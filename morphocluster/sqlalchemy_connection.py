import flask_sqlalchemy
from flask import g
from sqlalchemy.engine import Connection


class SQLAlchemyConnection(flask_sqlalchemy.SQLAlchemy):
    """
    Extend flask_sqlalchemy.SQLAlchemy by a connection.
    """

    def init_app(self, app):
        super().init_app(app)

        @app.teardown_appcontext
        def _(exc):
            connection = g.pop('sqlalchemy_connection', None)

            if connection is not None:
                connection.close()

            return exc

    def get_connection(self) -> Connection:
        if 'sqlalchemy_core_connection' not in g:
            g.sqlalchemy_core_connection = self.engine.connect()

        return g.sqlalchemy_core_connection
