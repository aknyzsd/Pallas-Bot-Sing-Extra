import os
from pathlib import Path
from pydantic import BaseModel, Extra
from nonebot import get_driver
from pyncm import apis as ncm
from src.common.utils.download_tools import DownloadTools
from nonebot import logger
import json  # 添加导入


class Config(BaseModel, extra=Extra.ignore):
    ncm_phone: str = ""
    ncm_email: str = ""
    ncm_password: str = ""
    ncm_ctcode: int = 86


try:
    # pydantic v2
    from nonebot import get_plugin_config
    config = get_plugin_config(Config)
except ImportError:
    # pydantic v1
    config = Config.parse_obj(get_driver().config)

if config.ncm_phone and config.ncm_password:
    ncm.login.LoginViaCellphone(
        phone=config.ncm_phone, password=config.ncm_password, ctcode=config.ncm_ctcode)
elif config.ncm_email and config.ncm_password:
    ncm.login.LoginViaEmail(email=config.ncm_email,
                            password=config.ncm_password)
else:
    ncm.login.LoginViaAnonymousAccount()


def download(song_id):
    folder = Path("resource/sing/ncm")
    path = folder / f"{song_id}.mp3"
    if path.exists():
        return path

    url = get_audio_url(song_id)
    if not url:
        return None

    content = request_file(url)
    if not content:
        return None

    os.makedirs(folder, exist_ok=True)
    with open(path, mode='wb+') as voice:
        voice.write(content)

    return path


def get_audio_url(song_id):
    try:
        response = ncm.track.GetTrackAudio(song_id)
        if isinstance(response, bytes):  # 如果返回值是字节类型，尝试解码为字符串
            response = response.decode("utf-8")
        if isinstance(response, str):  # 如果返回值是字符串，尝试解析为 JSON
            # 处理多条 JSON 数据的情况
            response = response.split("}{")
            if len(response) > 1:
                response = response[0] + "}"  # 提取第一条 JSON 数据
            response = json.loads(response)
        if not isinstance(response, dict):  # 确保返回值是字典
            logger.error(f"Invalid response format: {response}")
            return None
        if response.get("code") != 200:  # 检查返回的错误代码
            logger.error(f"API returned error: {response}")
            return None
        if "data" not in response or not response["data"]:
            logger.error(f"Missing 'data' in response: {response}")
            return None
        if response["data"][0]["size"] > 100000000:  # 100MB
            return None
        return response["data"][0]["url"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse response for song_id {song_id}: {e}")
        return None


def request_file(url):
    return DownloadTools.request_file(url)


def get_song_title(song_id):
    try:
        response = ncm.track.GetTrackDetail(song_id)
        if isinstance(response, bytes):
            response = response.decode("utf-8")
        if isinstance(response, str):
            # 处理多条 JSON 数据的情况
            response = response.split("}{")
            if len(response) > 1:
                response = response[0] + "}"  # 提取第一条 JSON 数据
            response = json.loads(response)
        if not isinstance(response, dict):
            logger.error(f"Invalid response format: {response}")
            return None
        if response.get("code") != 200:  # 检查返回的错误代码
            logger.error(f"API returned error: {response}")
            return None
        return response["songs"][0]["name"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse response for song_id {song_id}: {e}")
        return None


def get_song_id(song_name: str):
    if not song_name:
        return None
    try:
        res = ncm.cloudsearch.GetSearchResult(song_name, 1, 10)
        if isinstance(res, bytes):
            res = res.decode("utf-8")
        if isinstance(res, str):
            # 处理多条 JSON 数据的情况
            res = res.split("}{")
            if len(res) > 1:
                res = res[0] + "}"  # 提取第一条 JSON 数据
            res = json.loads(res)
        if not isinstance(res, dict):
            logger.error(f"Invalid response format: {res}")
            return None
        if res.get("code") != 200:  # 检查返回的错误代码
            logger.error(f"API returned error: {res}")
            return None
        if "result" not in res or "songCount" not in res["result"]:
            return None
        if res["result"]["songCount"] == 0:
            return None
        for song in res["result"]["songs"]:
            privilege = song["privilege"]
            if "chargeInfoList" not in privilege:
                continue
            charge_info_list = privilege["chargeInfoList"]
            if len(charge_info_list) == 0:
                continue
            if charge_info_list[0]["chargeType"] == 1:
                continue
            return song["id"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse response for song_name {song_name}: {e}")
        return None
    return None
