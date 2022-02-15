# import pemfc_dash
# import pemfc_dash.main
from pemfc_dash.main import app
server = app.server

if __name__ == "__main__":
    # [print(num, x) for num, x in enumerate(dl.ID_LIST) ]
    app.run_server(debug=True, use_reloader=False)
    # app.run_server(debug=True, use_reloader=False,
    #                host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

    