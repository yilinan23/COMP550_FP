"""Small command-line entry point for scaffold sanity checks."""

from syntax_rl.utils import configure_logging, get_logger, project_root, seed_everything


def main() -> None:
    """Run a minimal scaffold health check."""
    configure_logging()
    seed_state = seed_everything(42)
    logger = get_logger(__name__)
    logger.info("Syntax RL scaffold ready at %s with seed %s", project_root(), seed_state.seed)


if __name__ == "__main__":
    main()
