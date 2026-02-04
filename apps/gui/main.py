def main():
    print("MAIN: starting QApplication", flush=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        print("MAIN: constructing MainWindow", flush=True)
        window = MainWindow()
        print("MAIN: showing window", flush=True)
        window.show()
        print("MAIN: entering event loop", flush=True)
        rc = app.exec_()
        print("MAIN: event loop returned", rc, flush=True)
        sys.exit(rc)
    except Exception as e:
        import traceback
        print("MAIN: exception occurred:", e, flush=True)
        traceback.print_exc()
        raise
