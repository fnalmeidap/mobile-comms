from mobcom.io.reader import Reader

class DataProvider:
    @staticmethod
    def select(data: str):
        match data.lower():
            case "mnist":
                return Reader.download_data()
            case "default":
                return Reader.default_data()