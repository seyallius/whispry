//! Locate cached Whisper models or download them.
//!
//! Models are stored as plain files in `<cache>/<model name>/`, the same
//! layout earlier versions used, so models downloaded before are found.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::config::model_repository;
use crate::events::{Event, EventSink, Stage};
use crate::platform::Os;

/// Files fetched from the model repository. `.*` matches any extension.
pub const MODEL_FILES: &[&str] = &[
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "preprocessor_config.json",
];

pub const REQUIRED_FILES: &[&str] = &["config.json", "model.bin", "tokenizer.json"];

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Progress is reported at most once per this many bytes.
const PROGRESS_STEP: u64 = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Unknown model: {0}")]
    Unknown(String),
    #[error("Could not download the model: {0}")]
    Network(String),
    #[error("Could not save the model: {0}")]
    Io(#[from] std::io::Error),
    #[error("The model repository is missing {0}")]
    Incomplete(String),
    #[error("Download cancelled")]
    Cancelled,
}

/// Where models are cached for this user, following each OS's convention.
///
/// `WHISPRY_MODEL_DIR` overrides it.
pub fn default_model_directory() -> PathBuf {
    model_directory_for(
        Os::current(),
        |name| std::env::var_os(name),
        std::env::home_dir(),
    )
}

fn model_directory_for(
    os: Os,
    env: impl Fn(&str) -> Option<OsString>,
    home: Option<PathBuf>,
) -> PathBuf {
    if let Some(path) = env("WHISPRY_MODEL_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(path);
    }

    let home = home.unwrap_or_else(|| PathBuf::from("."));
    let base = match os {
        Os::Windows => env("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Local")),
        Os::MacOs => home.join("Library").join("Caches"),
        Os::Linux | Os::Other => env("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".cache")),
    };
    base.join("whispry").join("models")
}

/// A `models` folder beside the executable, for portable installs.
fn bundled_model_directory() -> Option<PathBuf> {
    Some(std::env::current_exe().ok()?.parent()?.join("models"))
}

pub fn is_model_cached(model: &str, directory: &Path) -> bool {
    let destination = directory.join(model);
    REQUIRED_FILES
        .iter()
        .all(|name| destination.join(name).is_file())
}

/// The directory of an already-downloaded model, if any.
pub fn find_cached_model(model: &str, directory: Option<&Path>) -> Option<PathBuf> {
    let mut candidates = vec![directory.map_or_else(default_model_directory, Path::to_path_buf)];
    if directory.is_none() {
        candidates.extend(bundled_model_directory());
    }
    candidates
        .into_iter()
        .find(|candidate| is_model_cached(model, candidate))
        .map(|candidate| candidate.join(model))
}

fn wanted(name: &str) -> bool {
    MODEL_FILES
        .iter()
        .any(|pattern| match pattern.strip_suffix(".*") {
            Some(stem) => name
                .strip_prefix(stem)
                .is_some_and(|rest| rest.starts_with('.') && rest.len() > 1),
            None => *pattern == name,
        })
}

struct RemoteFile {
    name: String,
    size: Option<u64>,
}

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .build()
        .into()
}

fn network_error(error: impl std::fmt::Display) -> ModelError {
    ModelError::Network(error.to_string())
}

fn list_files(
    agent: &ureq::Agent,
    endpoint: &str,
    repository: &str,
) -> Result<Vec<RemoteFile>, ModelError> {
    let url = format!("{endpoint}/api/models/{repository}?blobs=true");
    let text = agent
        .get(&url)
        .call()
        .map_err(network_error)?
        .body_mut()
        .read_to_string()
        .map_err(network_error)?;
    let listing: serde_json::Value = serde_json::from_str(&text).map_err(network_error)?;

    let mut files: Vec<RemoteFile> = listing["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let name = entry["rfilename"].as_str()?;
            wanted(name).then(|| RemoteFile {
                name: name.to_string(),
                size: entry["size"].as_u64(),
            })
        })
        .collect();

    if let Some(missing) = REQUIRED_FILES
        .iter()
        .find(|required| !files.iter().any(|file| file.name == **required))
    {
        return Err(ModelError::Incomplete((*missing).to_string()));
    }

    // The big file goes last, so a model that has all required files is whole.
    files.sort_by_key(|file| file.name == "model.bin");
    Ok(files)
}

struct Download<'a> {
    on_event: EventSink<'a>,
    cancel: &'a CancelToken,
    total: Option<f64>,
    completed: u64,
    reported: u64,
}

impl Download<'_> {
    fn report(&mut self, force: bool) {
        if force || self.completed - self.reported >= PROGRESS_STEP {
            self.reported = self.completed;
            (self.on_event)(Event::Progress {
                stage: Stage::Model,
                completed: self.completed as f64,
                total: self.total,
                unit: "B".into(),
            });
        }
    }

    fn fetch(
        &mut self,
        agent: &ureq::Agent,
        url: &str,
        destination: &Path,
    ) -> Result<(), ModelError> {
        let mut response = agent.get(url).call().map_err(network_error)?;
        let partial = destination.with_extension("part");
        let result = self.copy(&mut response.body_mut().as_reader(), &partial);
        match result {
            Ok(()) => Ok(std::fs::rename(&partial, destination)?),
            Err(error) => {
                let _ = std::fs::remove_file(&partial);
                Err(error)
            }
        }
    }

    fn copy(&mut self, source: &mut dyn Read, partial: &Path) -> Result<(), ModelError> {
        let mut file = File::create(partial)?;
        let mut buffer = vec![0u8; 256 * 1024];
        loop {
            if self.cancel.is_cancelled() {
                return Err(ModelError::Cancelled);
            }
            let count = source.read(&mut buffer).map_err(network_error)?;
            if count == 0 {
                return Ok(file.flush()?);
            }
            file.write_all(&buffer[..count])?;
            self.completed += count as u64;
            self.report(false);
        }
    }
}

fn download_model(
    agent: &ureq::Agent,
    endpoint: &str,
    repository: &str,
    destination: &Path,
    on_event: EventSink,
    cancel: &CancelToken,
) -> Result<(), ModelError> {
    let files = list_files(agent, endpoint, repository)?;
    std::fs::create_dir_all(destination)?;

    let sizes: Option<Vec<u64>> = files.iter().map(|file| file.size).collect();
    let mut download = Download {
        on_event,
        cancel,
        total: sizes.map(|sizes| sizes.iter().sum::<u64>() as f64),
        completed: 0,
        reported: 0,
    };
    download.report(true);

    for file in &files {
        let url = format!("{endpoint}/{repository}/resolve/main/{}", file.name);
        download.fetch(agent, &url, &destination.join(&file.name))?;
    }
    download.report(true);
    Ok(())
}

/// The local model directory, downloading the model if it is missing.
pub fn ensure_model(
    model: &str,
    on_event: EventSink,
    cancel: &CancelToken,
    directory: Option<&Path>,
) -> Result<PathBuf, ModelError> {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
    ensure_model_from(&endpoint, &http_agent(), model, on_event, cancel, directory)
}

fn ensure_model_from(
    endpoint: &str,
    agent: &ureq::Agent,
    model: &str,
    on_event: EventSink,
    cancel: &CancelToken,
    directory: Option<&Path>,
) -> Result<PathBuf, ModelError> {
    let repository = model_repository(model).ok_or_else(|| ModelError::Unknown(model.into()))?;

    if let Some(cached) = find_cached_model(model, directory) {
        on_event(Event::Log(format!(
            "Using cached model: {}",
            cached.display()
        )));
        return Ok(cached);
    }

    let destination = directory
        .map_or_else(default_model_directory, Path::to_path_buf)
        .join(model);
    on_event(Event::Log(format!(
        "Downloading model {model} to {}",
        destination.display()
    )));
    download_model(
        agent,
        endpoint.trim_end_matches('/'),
        repository,
        &destination,
        on_event,
        cancel,
    )?;
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn fake_model(directory: &Path, name: &str) {
        let target = directory.join(name);
        std::fs::create_dir_all(&target).unwrap();
        for file in REQUIRED_FILES {
            std::fs::write(target.join(file), "x").unwrap();
        }
    }

    #[test]
    fn the_environment_variable_overrides_the_cache_folder() {
        let env = |name: &str| (name == "WHISPRY_MODEL_DIR").then(|| OsString::from("/models"));
        assert_eq!(
            model_directory_for(Os::Linux, env, None),
            PathBuf::from("/models")
        );
    }

    #[test]
    fn cache_folders_follow_each_os() {
        let home = Some(PathBuf::from("/home/me"));
        let none = |_: &str| None;
        assert_eq!(
            model_directory_for(Os::Linux, none, home.clone()),
            PathBuf::from("/home/me/.cache/whispry/models")
        );
        assert_eq!(
            model_directory_for(Os::MacOs, none, home.clone()),
            PathBuf::from("/home/me/Library/Caches/whispry/models")
        );
        let windows = |name: &str| (name == "LOCALAPPDATA").then(|| OsString::from("C:\\Local"));
        assert_eq!(
            model_directory_for(Os::Windows, windows, home),
            PathBuf::from("C:\\Local").join("whispry").join("models")
        );
        let xdg = |name: &str| (name == "XDG_CACHE_HOME").then(|| OsString::from("/xdg"));
        assert_eq!(
            model_directory_for(Os::Linux, xdg, None),
            PathBuf::from("/xdg/whispry/models")
        );
    }

    #[test]
    fn finds_a_cached_model() {
        let dir = tempfile::tempdir().unwrap();
        fake_model(dir.path(), "small");
        assert_eq!(
            find_cached_model("small", Some(dir.path())),
            Some(dir.path().join("small"))
        );
        assert_eq!(find_cached_model("medium", Some(dir.path())), None);
    }

    #[test]
    fn a_model_missing_a_required_file_is_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        fake_model(dir.path(), "small");
        std::fs::remove_file(dir.path().join("small/model.bin")).unwrap();
        assert!(!is_model_cached("small", dir.path()));
    }

    #[test]
    fn picks_only_the_files_a_model_needs() {
        for name in [
            "config.json",
            "model.bin",
            "vocabulary.txt",
            "vocabulary.json",
        ] {
            assert!(wanted(name), "{name}");
        }
        for name in [
            "README.md",
            "vocabulary",
            "vocabulary.",
            "sub/model.bin",
            "model.bin.part",
        ] {
            assert!(!wanted(name), "{name}");
        }
    }

    /// Serves `routes` over plain HTTP on a free local port.
    struct Server {
        endpoint: String,
        requests: Arc<AtomicUsize>,
    }

    fn serve(routes: HashMap<String, Vec<u8>>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let routes = Arc::new(Mutex::new(routes));
        let counter = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                counter.fetch_add(1, Ordering::SeqCst);
                let routes = routes.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(&stream);
                    let mut request_line = String::new();
                    reader.read_line(&mut request_line).unwrap();
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).unwrap();
                        if header.trim().is_empty() {
                            break;
                        }
                    }
                    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
                    let path = path.split('?').next().unwrap();
                    let body = routes.lock().unwrap().get(path).cloned();
                    let mut stream = &stream;
                    match body {
                        // A body that starts with this marker is a redirect.
                        Some(body) if body.starts_with(b"REDIRECT ") => {
                            let target = String::from_utf8_lossy(&body[9..]).into_owned();
                            let head = format!(
                                "HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            );
                            let _ = stream.write_all(head.as_bytes());
                        }
                        Some(body) => {
                            let head = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = stream.write_all(head.as_bytes());
                            let _ = stream.write_all(&body);
                        }
                        None => {
                            let _ = stream.write_all(
                                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            );
                        }
                    }
                });
            }
        });
        Server { endpoint, requests }
    }

    fn repository_routes(extra_files: &[(&str, usize)]) -> HashMap<String, Vec<u8>> {
        let files: Vec<(&str, usize)> = [
            ("config.json", 20),
            ("model.bin", 3 << 20),
            ("tokenizer.json", 30),
            ("vocabulary.txt", 40),
            ("README.md", 50),
        ]
        .into_iter()
        .chain(extra_files.iter().copied())
        .collect();

        let siblings: Vec<serde_json::Value> = files
            .iter()
            .map(|(name, size)| serde_json::json!({"rfilename": name, "size": size}))
            .collect();
        let mut routes = HashMap::new();
        routes.insert(
            "/api/models/Systran/faster-whisper-small".to_string(),
            serde_json::json!({ "siblings": siblings })
                .to_string()
                .into_bytes(),
        );
        for (name, size) in files {
            routes.insert(
                format!("/Systran/faster-whisper-small/resolve/main/{name}"),
                vec![b'a'; size],
            );
        }
        routes
    }

    fn plain_agent() -> ureq::Agent {
        // Tests talk to localhost, so ignore any proxy from the environment.
        ureq::Agent::config_builder().proxy(None).build().into()
    }

    #[test]
    fn downloads_the_needed_files_then_reuses_them() {
        let server = serve(repository_routes(&[]));
        let cache = tempfile::tempdir().unwrap();
        let mut events = Vec::new();

        let path = ensure_model_from(
            &server.endpoint,
            &plain_agent(),
            "small",
            &mut |event| events.push(event),
            &CancelToken::new(),
            Some(cache.path()),
        )
        .unwrap();

        assert_eq!(path, cache.path().join("small"));
        let mut names: Vec<String> = std::fs::read_dir(&path)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.txt"
            ]
        );
        assert_eq!(
            std::fs::metadata(path.join("model.bin")).unwrap().len(),
            3 << 20
        );

        let progress: Vec<(f64, Option<f64>)> = events
            .iter()
            .filter_map(|event| match event {
                Event::Progress {
                    completed,
                    total,
                    unit,
                    ..
                } if unit == "B" => Some((*completed, *total)),
                _ => None,
            })
            .collect();
        let total = (20 + 30 + 40 + (3 << 20)) as f64;
        assert_eq!(progress.first(), Some(&(0.0, Some(total))));
        assert_eq!(progress.last(), Some(&(total, Some(total))));
        assert!(
            progress.len() > 3,
            "progress should be reported along the way"
        );

        let requests = server.requests.load(Ordering::SeqCst);
        let mut events = Vec::new();
        ensure_model_from(
            &server.endpoint,
            &plain_agent(),
            "small",
            &mut |event| events.push(event),
            &CancelToken::new(),
            Some(cache.path()),
        )
        .unwrap();
        assert_eq!(server.requests.load(Ordering::SeqCst), requests);
        assert!(matches!(&events[0], Event::Log(text) if text.contains("Using cached")));
    }

    #[test]
    fn cancelling_stops_the_download_and_leaves_no_partial_file() {
        let server = serve(repository_routes(&[]));
        let cache = tempfile::tempdir().unwrap();
        let cancel = CancelToken::new();
        let trigger = cancel.clone();

        let result = ensure_model_from(
            &server.endpoint,
            &plain_agent(),
            "small",
            &mut |event| {
                if let Event::Progress { completed, .. } = event
                    && completed > (1 << 20) as f64
                {
                    trigger.cancel();
                }
            },
            &cancel,
            Some(cache.path()),
        );

        assert!(matches!(result, Err(ModelError::Cancelled)));
        let leftovers: Vec<_> = std::fs::read_dir(cache.path().join("small"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert!(
            !leftovers.iter().any(|name| name.ends_with(".part")),
            "{leftovers:?}"
        );
        assert!(!is_model_cached("small", cache.path()));
    }

    #[test]
    fn follows_the_redirect_hugging_face_sends_to_its_cdn() {
        let mut routes = repository_routes(&[]);
        // The real site answers a file request with a 302 to another server.
        let big = routes
            .remove("/Systran/faster-whisper-small/resolve/main/model.bin")
            .unwrap();
        routes.insert("/cdn/abc123".into(), big);
        routes.insert(
            "/Systran/faster-whisper-small/resolve/main/model.bin".into(),
            b"REDIRECT /cdn/abc123".to_vec(),
        );
        let server = serve(routes);
        let cache = tempfile::tempdir().unwrap();

        let path = ensure_model_from(
            &server.endpoint,
            &plain_agent(),
            "small",
            &mut |_| {},
            &CancelToken::new(),
            Some(cache.path()),
        )
        .unwrap();
        assert_eq!(
            std::fs::metadata(path.join("model.bin")).unwrap().len(),
            3 << 20
        );
    }

    #[test]
    fn a_repository_without_a_required_file_is_reported() {
        let mut routes = repository_routes(&[]);
        let listing = serde_json::json!({"siblings": [{"rfilename": "config.json", "size": 20}]});
        routes.insert(
            "/api/models/Systran/faster-whisper-small".into(),
            listing.to_string().into_bytes(),
        );
        let server = serve(routes);
        let cache = tempfile::tempdir().unwrap();

        let result = ensure_model_from(
            &server.endpoint,
            &plain_agent(),
            "small",
            &mut |_| {},
            &CancelToken::new(),
            Some(cache.path()),
        );
        assert!(matches!(result, Err(ModelError::Incomplete(name)) if name == "model.bin"));
    }

    #[test]
    fn a_failing_server_is_a_network_error_and_unknown_models_are_rejected() {
        let server = serve(HashMap::new());
        let cache = tempfile::tempdir().unwrap();
        let ensure = |model: &str| {
            ensure_model_from(
                &server.endpoint,
                &plain_agent(),
                model,
                &mut |_| {},
                &CancelToken::new(),
                Some(cache.path()),
            )
        };
        assert!(matches!(ensure("small"), Err(ModelError::Network(_))));
        assert!(matches!(ensure("huge"), Err(ModelError::Unknown(_))));
    }
}
