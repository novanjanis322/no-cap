"""Unit tests for CLI module."""

from click.testing import CliRunner

from bsort.cli import cli


class TestCLI:
    """Test cases for CLI commands."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Bottle cap color detection CLI" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_train_command_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train the bottle cap detection model" in result.output

    def test_infer_command_help(self):
        """Test infer command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["infer", "--help"])
        assert result.exit_code == 0
        assert "Run inference on an image" in result.output

    def test_train_command_missing_config(self):
        """Test train command without config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0

    def test_infer_command_missing_args(self):
        """Test infer command without required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["infer"])
        assert result.exit_code != 0
