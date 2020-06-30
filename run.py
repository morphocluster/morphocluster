"""
Run the MorphoCluster application.

This file is only for debug purposes.
In a production environment, the application has to be started using e.g. gunicorn.
"""
from morphocluster import create_app

if __name__ == "__main__":
    app = create_app()
    app.run()
