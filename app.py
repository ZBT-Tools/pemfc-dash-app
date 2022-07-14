# import pemfc_dash
# import pemfc_dash.main
from pemfc_dash.main import app #, celery_app
server = app.server
# celery_app = celery_app
if __name__ == "__main__":
    # [print(num, x) for num, x in enumerate(dl.ID_LIST) ]
    app.run_server(debug=True, use_reloader=False)
    # app.run_server(debug=True, use_reloader=False,
    #                host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
