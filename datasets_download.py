import os
import gdown


#Identifiers the respective datasets folders on Google Disk
file_ids = {
    'autotrain_TG_sentiment': '1fs91bOR9PCTwT3grmXd26Bq4G5KcIWpm',
    'truw': '1IcOxaDMeawj9j5W0PYN-OCm11gw3AuSr',
    'cyberpol': '1BxrmeMyJ-ld1KV5gkGm2EnrvaE3Dmg8H',
    'kyiv_digital_sentiment_annotation': '1RXjgI8u7A6_FsjI3O52sMLS8X0gI8K7P',
    'emotions_sentiment_youscan': '1gP8weiLFJyyzyBzL1-TvO-fbHpRap-mv',
    'dataset_5-1': '1Xu_1hh80gjSxLN4rgQ6dhqHfZjlRRD9E',
    'yahoo_reviews': '1RUICT82eZIMxMx93zxqohhG0dJosdoVW',
    'telegram_disinformation': '1l_sc8JplNAx2dPoN75o9EhZF_oarRcAl'
}


destination_folder = 'data_provided'  #added to .gitignore 



def ensure_directory_exists(directory):
    """
    It ensures that the specified directory exists.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created folder: {directory}")
    else:
        print(f"Folder already exists: {directory}")



def download_datasets(file_ids, destination_folder):
    """
    Downloads datasets using gdown.
    """

    ensure_directory_exists(destination_folder)

    for dataset_name, file_id in file_ids.items():
        print(f"Downloading {dataset_name}...")
        destination_path = os.path.join(destination_folder, f"{dataset_name}.zip")
        try:
            gdown.download_folder(f"https://drive.google.com/drive/folders/{file_id}", quiet=False, output=destination_path)
            print(f"Downloaded {dataset_name} to {destination_path}")
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")





if __name__ == "__main__":
    print("Starting dataset download...")
    download_datasets(file_ids, destination_folder)
    print("All datasets have been processed.")