
# import dash_bootstrap_components as dbc
# from dash.long_callback import CeleryLongCallbackManager, \
#     DiskcacheLongCallbackManager
import os
import redis
import shutil
from dash_extensions.enrich import DashProxy, MultiplexerTransform, \
    ServersideOutputTransform, RedisBackend, FileSystemBackend


def clear_cache(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def make_file_system_store(file_path='temp/file_system_store'):
    store_dir = os.path.join(os.getcwd(), 'temp/file_system_store')
    clear_cache(store_dir)
    return FileSystemBackend(cache_dir=store_dir)


try:
    import pemfc_dash.redis_credentials as rc
    caching_backend = RedisBackend(
        host=rc.HOST_NAME,
        password=rc.PASSWORD,
        port=rc.PORT,
        default_timeout=900)
    try:
        caching_backend.delete('test')
    except (redis.exceptions.ConnectionError, ConnectionRefusedError) as E:
        caching_backend = make_file_system_store()
    except (redis.exceptions.ResponseError, redis.exceptions.RedisError):
        pass
except (ImportError, AttributeError) as E:
    caching_backend = RedisBackend(host='redis', port=6379)
    try:
        caching_backend.delete('test')
    except (redis.exceptions.ConnectionError, ConnectionRefusedError, ImportError) as E:
        caching_backend = make_file_system_store()


# from celery import Celery
# import diskcache

# cache = diskcache.Cache("./cache")
# long_callback_manager = DiskcacheLongCallbackManager(cache)

# celery_app = Celery(
#     __name__, broker="redis://localhost:6379/0",
#     backend="redis://localhost:6379/1")
# long_callback_manager = CeleryLongCallbackManager(celery_app)

# app = dash.Dash(__name__, long_callback_manager=long_callback_manager)

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
bs_4_css = ('https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css'
              '/010_bootstrap.min.css')
bs_5_css = ('https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css')

external_stylesheets = [bs_5_css]
app = DashProxy(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True,
                transforms=[MultiplexerTransform(),
                            ServersideOutputTransform(backends=[caching_backend])])

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
#                 long_callback_manager=long_callback_manager,
#                 suppress_callback_exceptions=True)
# app = dash.Dash(__name__, suppress_callback_exceptions=True)
# server = app.server
