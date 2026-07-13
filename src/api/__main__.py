import uvicorn


def main():
    """
    Run the AdaptiveRAG API application.
    """
    uvicorn.run(
        "src.api.app:create_app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload for development
        factory=True,  # Use factory mode to create the app instance
    )


if __name__ == "__main__":
    main()
