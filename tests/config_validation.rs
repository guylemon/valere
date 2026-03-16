use std::process::Command;

#[test]
fn test_example_config_is_valid() {
    let output = Command::new("cargo")
        .args(["run", "--", "--validate", "--config", "valere.toml.example"])
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Validation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
