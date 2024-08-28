from modules.app_config import AppConfig
from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient


def get_box_client() -> BoxClient:
    app_config = AppConfig()
    ccg_config = CCGConfig(
        client_id=app_config.get("BOX_CLIENT_ID"),
        client_secret=app_config.get("BOX_CLIENT_SECRET"),
        user_id=app_config.get("BOX_USER_ID"),
    )
    ccg_auth = BoxCCGAuth(ccg_config)
    return BoxClient(ccg_auth)
