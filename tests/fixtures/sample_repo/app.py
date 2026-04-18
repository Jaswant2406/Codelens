def helper(name: str) -> str:
    """Create a greeting."""
    return f"Hello {name}"


def greet(user: str) -> str:
    """Public entry point."""
    message = helper(user)
    return message.upper()


def unused() -> None:
    pass
