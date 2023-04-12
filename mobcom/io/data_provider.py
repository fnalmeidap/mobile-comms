from mobcom.io.reader import Reader

class DataProvider:
    def select(self, data: str):
        match data.lower():
            case "mnist":
                return Reader.download_data()
            case "default":
                return Reader.default_data()