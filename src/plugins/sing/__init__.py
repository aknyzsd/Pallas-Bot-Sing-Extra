import os  # 添加导入

# 设置 PYTORCH_CUDA_ALLOC_CONF 以减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import torch  # 确保在设置环境变量后导入 PyTorch
from pathlib import Path
from threading import Lock
from asyncer import asyncify
import random
import time
import wave
import contextlib
from nonebot import on_message, require, logger
from nonebot.typing import T_State
from nonebot.rule import Rule
from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import Bot, MessageSegment, Message, permission, GroupMessageEvent, MessageEvent  # 添加导入
from nonebot.permission import SUPERUSER
from pydub.utils import mediainfo
from nonebot.params import CommandArg
from nonebot.plugin import on_command
from nonebot.exception import FinishedException  # 添加导入
from nonebot.internal.matcher import Matcher  # 添加导入
from torch import cuda  # 添加导入  # 添加导入

from src.common.config import GroupConfig, plugin_config

from .ncm_loader import download, get_song_title, get_song_id
from .slicer import slice
from .mixer import mix, splice
from .separater import separate, set_separate_cuda_devices
from .svc_inference import inference, set_svc_cuda_devices

if plugin_config.sing_cuda_device:
    set_separate_cuda_devices(plugin_config.sing_cuda_device)
    set_svc_cuda_devices(plugin_config.sing_cuda_device)

require("src.plugins.nonebot_plugin_gocqhttp_cross_machine_upload_file")
require("src.plugins.custom_face")
from src.plugins.nonebot_plugin_gocqhttp_cross_machine_upload_file import upload_file
from src.plugins.custom_face import fetch_custom_face_list, send_custom_face, update_custom_face_list

#custom_face_list = await asyncify(get_custom_face_cmd)


SING_CMD = '唱歌'
SING_CONTINUE_CMDS = ['继续唱', '接着唱']
SING_COOLDOWN_KEY = 'sing'


async def is_to_sing(bot: Bot, event: Event, state: T_State) -> bool:
    text = event.get_plaintext()
    if not text:
        return False
    
    if not SING_CMD in text and not any([cmd in text for cmd in SING_CONTINUE_CMDS]):
        return False
    
    if text.endswith(SING_CMD):
        return False

    has_spk = False
    for name, speaker in plugin_config.sing_speakers.items():
        if not text.startswith(name):
            continue
        text = text.replace(name, '').strip()
        has_spk = True
        state['speaker'] = speaker
        state['failed_speaker_name'] = name
        state['speaker_name_msg'] = name
        break

    if not has_spk:
        return False

    if "key=" in text or "-k " in text:
        if "key=" in text:
            key_pos = text.find("key=")
            key_val = text[key_pos + 4:].strip()  # 获取 key= 后面的值
            text = text.replace("key=" + key_val, "")  # 去掉消息中的 key 信息
        else:
            key_pos = text.find("-k ")
            key_val = text[key_pos + 3:].split()[0]  # 获取 -k 后面的值
            text = text.replace(f"-k {key_val}", "").strip()  # 去掉消息中的 -k 信息
        try:
            key_int = int(key_val)  # 判断输入的 key 是不是整数
            if key_int < -12 or key_int > 12:
                return False  # 限制一下 key 的大小，一个八度应该够了
        except ValueError:
            return False
    else:
        key_val = 0
    state['key'] = key_val


    # 解析 -t 参数
    if "-t " in text:
        t_pos = text.find("-t ")
        t_val = text[t_pos + 3:].split()[0]  # 获取 -t 后面的值
        text = text.replace(f"-t {t_val}", "").strip()  # 去掉消息中的 -t 信息
        try:
            t_int = int(t_val)  # 判断输入的时长是否为整数
            if t_int <= 0:
                return False  # 时长必须为正数
        except ValueError:
            return False
    else:
        t_int = None
    state['duration'] = t_int
    state['duration2'] = t_int
    state['use_t_mode'] = True

    # 解析 -s 参数
    if "-s " in text or "--source " in text:
        if "-s " in text:
            s_pos = text.find("-s ")
            source_val = text[s_pos + 3:].split()[0]  # 获取 -s 后面的值
            text = text.replace(f"-s {source_val}", "").strip()  # 去掉消息中的 -s 信息
        else:
            s_pos = text.find("--source ")
            source_val = text[s_pos + 9:].split()[0]  # 获取 --source 后面的值
            text = text.replace(f"--source {source_val}", "").strip()  # 去掉消息中的 --source 信息

        if source_val not in ["ncm", "local"]:
            await bot.send(event, "歌曲源只能从ncm和local中选啊喂！")  # 发送提示信息
            return False  # 如果 source 参数值无效，则返回 False
            
    else:
        source_val = "local"  # 默认值为 local
    state['source'] = source_val

    if "--soyo--force" in text:
        state['soyo_force'] = True

    if text.startswith(SING_CMD):
        song_key = text.replace(SING_CMD, '').strip()
        if not song_key:
            return False
        state['song_id'] = song_key
        state['chunk_index'] = 0
        return True

    if text in SING_CONTINUE_CMDS:
        progress = GroupConfig(group_id=event.group_id).sing_progress()
        if not progress:
            logger.error(f"No progress found for group_id: {event.group_id}")
            return False

        song_id = progress['song_id']
        song_id2 = progress['song_id']
        chunk_index = progress['chunk_index']
        key_val = progress['key']
        logger.info(f"Continuing song with song_id: {song_id}, chunk_index: {chunk_index}, key: {key_val}")

        if not song_id or chunk_index > 100:
            logger.error(f"Invalid song_id or chunk_index out of range: song_id={song_id}, chunk_index={chunk_index}")
            return False

        # 添加日志以检查 song_id 和文件路径的生成
        expected_file_path = Path(f"resource/sing/slices/{song_id}_chunk{chunk_index}.mp3")
        logger.debug(f"Expected file path: {expected_file_path.resolve()}")

        state['song_id'] = str(song_id)
        state['continue_song_id'] = song_id2
        # 不知道为什么，继续唱的时候get_local_song会错误的把song_id作为歌曲名再请求一遍，
        # 所以如果是继续唱的话就请求两个song_id，把那个正常的song_id送进bug里
        # 把真正需要的song_id绕过get_local_song，也就是song_id2
        # 检测到use_existing_song_id = True时就使用song_id2
        # 变量名乱取的，毕竟也没想深入开发，就这样吧
        state['chunk_index'] = chunk_index
        state['key'] = key_val
        state['source'] = progress.get('source', 'local')  # 获取 source 信息，默认为 local
        state['use_existing_song_id'] = True  # 标记直接使用现有 song_id
        logger.info(f"继续唱日志: song_id: {song_id}")
        logger.info(f"继续唱日志: song_id2: {song_id2}")
        return True

    return False

sing_msg = on_message(
    rule=Rule(is_to_sing),
    priority=5,
    block=True,
    permission=permission.GROUP
)

gpu_locker = Lock()

LOCAL_MUSIC_PATH = 'resource/local_music/'  # 定义本地歌曲库路径

async def get_local_song(song_name: str) -> Path:
    """
    从本地歌曲库中查找歌曲文件
    """
    if not os.path.exists(LOCAL_MUSIC_PATH):
        return None

    for file_path in Path(LOCAL_MUSIC_PATH).glob("*.mp3"):
        if song_name.lower() in file_path.stem.lower():
            return file_path
    return None


@sing_msg.handle()
async def _(bot: Bot, event: GroupMessageEvent, state: T_State):
    config = GroupConfig(event.group_id, cooldown=120)
    if not config.is_cooldown(SING_COOLDOWN_KEY):
        return
    config.refresh_cooldown(SING_COOLDOWN_KEY)

    speaker = state['speaker']
    song_key = state['song_id']
    chunk_index = state['chunk_index']
    key = state['key']
    duration = state.get('duration', None)
    duration2 = state.get('duration2', None)
    failed_speaker_name = state['failed_speaker_name']
    speaker_name_msg = state['speaker_name_msg']
    if duration is None:
        duration = plugin_config.sing_length  # 使用 .env 文件中的默认时长

    source = state.get('source', 'local')  # 获取 source 参数值，默认为 local

    async def failed(error_message=None):
        config.reset_cooldown(SING_COOLDOWN_KEY)
        if error_message and "OutOfMemoryError" in error_message:
            await sing_msg.finish('寄，爆显存了喵，不要把这么长的原曲塞进来啊！尝试减少歌曲长度或降低模型复杂度。')
        elif error_message and "get_song_id failed" in error_message:
            await sing_msg.finish('请求太多被网易ban了喵，给我等一会啊！')
        else:
            await sing_msg.finish('寄，这次没唱好喵😭')

    async def svc_failed():
        config.reset_cooldown(SING_COOLDOWN_KEY)
        svc_failed_msg = f"""好长的歌喵，{failed_speaker_name}唱不动了，下次让我唱少一点吧😔"""
        await sing_msg.finish(svc_failed_msg)
    
    async def download_failed():
        config.reset_cooldown(SING_COOLDOWN_KEY)
        await sing_msg.finish('下载或向网易云查询歌曲失败喵，可能短时间内请求太多次了被网易云ban了喵，给我等一会啊！')

    async def separated_failed():
        config.reset_cooldown(SING_COOLDOWN_KEY)
        separated_failed_msg = f"""寄，人声分离失败了喵！{failed_speaker_name}我啊，也不知道为什么呢"""
        await sing_msg.finish(separated_failed_msg)

    async def success(song: Path, spec_index: int = None):
        config.reset_cooldown(SING_COOLDOWN_KEY)
        config.update_sing_progress({
            'song_id': song_id,
            'chunk_index': (spec_index if spec_index else chunk_index) + 1,
            'key': key,
        })
        with open((song), 'rb') as f:    
            data = f.read()
        msg: Message = MessageSegment.record(file=data)
        await sing_msg.finish(msg)
    
    ##先刷一下自定义表情
    logger_update_face = await update_custom_face_list(bot)
    logger.info(f"更新自定义表情列表成功，获取到 {logger_update_face} 个自定义表情")


    try:
        # 下载 -> 切片 -> 人声分离 -> 音色转换（SVC） -> 混音
        if speaker_name_msg == "猫雷":
            await sing_msg.send('喵喵露们，聆听圣猫雷的福音罢！')
        elif speaker_name_msg == "柏林以东":
            await sing_msg.send('你会是个勇敢的发声者吗……')
        elif speaker_name_msg == "37":
            await sing_msg.send('欢迎来到数的世界。')
        elif speaker_name_msg == "星瞳":
            await sing_msg.send('小星星们早上中午晚上好呀！')
        elif speaker_name_msg == "塔菲":
            await sing_msg.send('塔菲最喜欢雏草姬了喵！')
        elif speaker_name_msg == "小菲":
            await sing_msg.send('可恶的大菲，又让我演奏😭')
        elif speaker_name_msg in ['soyo', '素世', '素食', '爽世']:
            #await sing_msg.send('为什么要演奏《春日影》？！！！！！')
            soyo_face_id = 'face_5'
            await send_custom_face(bot, event, soyo_face_id)
            if state.get('soyo_force', False) or song_key not in ['春日影']:
                await sing_msg.send('好吧，我就勉强演奏一下吧！')
            else:
                await failed()
        elif speaker_name_msg == "银狼":
            await sing_msg.send('今天也上线啦?')
        else:
            await sing_msg.send('欢呼吧！')

        # 优先从本地歌曲库中查找歌曲
        if source == "local":
            local_song, local_song_id = await get_local_song_with_id(song_key)
            if local_song:
                logger.info(f"Found local song: {local_song} with song_id: {local_song_id}")
                origin = local_song
                song_id = local_song_id  # 使用 local_music 的 song_id
            else:
                # 尝试将 song_key 当作 local_music 的 song_id 查找
                local_song_by_id = None
                for file_path, s_id in local_music_ids.items():
                    if s_id == song_key:
                        local_song_by_id = Path(file_path)
                        local_song_id = s_id
                        break
                
                if local_song_by_id:
                    logger.info(f"Found local song by id: {local_song_by_id} with song_id: {local_song_id}")
                    origin = local_song_by_id
                    song_id = local_song_id
                else:
                    # 如果本地未找到，则尝试从 ncm 下载
                    if state.get('use_existing_song_id', False):
                    # 如果标记为直接使用现有 song_id，则跳过 get_song_id
                        song_id = state['continue_song_id']
                        logger.info(f"继续唱直接使用现有 song_id: {song_id}")
                    else:
                        song_id = await asyncify(get_song_id)(song_key)
                        logger.info(f"下载日志:song_id: {song_id}")
                    logger.info(f"下载日志:song_id: {song_id}")
                    ncm_cache_path = Path(f'resource/sing/ncm/{song_id}.mp3')
                    if ncm_cache_path.exists():
                        logger.info(f"Found cached song in ncm: {ncm_cache_path}")
                        origin = ncm_cache_path
                    else:
                        origin = await asyncify(download)(song_id)
                        if not origin:
                            logger.error('download failed', song_id)
                            await download_failed()
        elif source == "ncm":
            # 强制从网易云下载
            if state.get('use_existing_song_id', False):
                # 如果标记为直接使用现有 song_id，则跳过 get_song_id
                song_id = state['continue_song_id']
                logger.info(f"继续唱直接使用现有 song_id: {song_id}")
            else:
                song_id = await asyncify(get_song_id)(song_key)
                logger.info(f"下载日志:song_id: {song_id}")

            # 检查本地 ncm 缓存是否存在
            ncm_cache_path = Path(f'resource/sing/ncm/{song_id}.mp3')
            if ncm_cache_path.exists():
                logger.info(f"Found cached song in ncm: {ncm_cache_path}")
                origin = ncm_cache_path
            else:
                origin = await asyncify(download)(song_id)
                if not origin:
                    logger.error('download failed', song_id)
                    await download_failed()
        
        # 获取歌曲总时长
        total_duration = await asyncify(get_song_duration)(origin)
        if not total_duration:
            logger.error('failed to get song duration', song_id)
            await failed()

        # 如果 -t 参数超出歌曲长度，按歌曲长度处理
        if duration > total_duration:
            logger.info(f"Provided duration ({duration}s) exceeds song length ({total_duration}s). Using song length.")
            duration = total_duration

        if chunk_index == 0:
            # 这里应该可以解耦成在mixer.py里处理，但是懒得动了，一块在这写吧
            # 要是不删掉缓存mixer.py就会直接返回原来的缓存，不会重新将svc文件夹里的推理好的新的时长的歌曲混音
            # 挺大力出奇迹的
            for cache_path in Path('resource/sing/splices').glob(f'{song_id}_*_{key}key_{speaker}.mp3'):
                if cache_path.name.startswith(f'{song_id}_full_'):
                    cache_duration = await asyncify(get_song_duration)(cache_path)
                    if duration and duration > cache_duration:
                        logger.info(f"Duration {duration}s exceeds cache duration {cache_duration}s. Deleting cache and reprocessing.")
                        # 删除 splices 和 mix 以及 svc 文件夹中的相关缓存文件
                        # 有的时候svc文件夹里有缓存不会推理，但有的时候又会推理，总之还是都删了比较好
                        for folder in ['splices', 'mix', 'svc']:
                            folder_path = Path(f'resource/sing/{folder}')
                            for file in folder_path.glob(f'{song_id}_*_{key}key_{speaker}*.*'):
                                try:
                                    file.unlink()
                                    logger.info(f"Deleted cache file: {file.resolve()}")
                                except Exception as e:
                                    logger.error(f"Failed to delete cache file {file.resolve()}: {e}")
                    else:
                        await success(cache_path, 114514)
                elif cache_path.name.startswith(f'{song_id}_spliced'):
                    cache_duration = await asyncify(get_song_duration)(cache_path)
                    if duration and duration > cache_duration:
                        logger.info(f"Duration {duration}s exceeds cache duration {cache_duration}s. Deleting cache and reprocessing.")
                        # 删除 splices 和 mix 文件夹中的相关缓存文件
                        for folder in ['splices', 'mix', 'svc']:
                            folder_path = Path(f'resource/sing/{folder}')
                            for file in folder_path.glob(f'{song_id}_*_{key}key_{speaker}*.*'):
                                try:
                                    file.unlink()
                                    logger.info(f"Deleted cache file: {file.resolve()}")
                                except Exception as e:
                                    logger.error(f"Failed to delete cache file {file.resolve()}: {e}")
                    else:
                        await success(cache_path, int(cache_path.name.split('_')[1].replace('spliced', '')))
        else:
            cache_path = Path("resource/sing/mix") / \
                f'{song_id}_chunk{chunk_index}_{key}key_{speaker}.mp3'
            if cache_path.exists():
                await asyncify(splice)(cache_path, Path('resource/sing/splices'), False, song_id, chunk_index, speaker, key=key)
                await success(cache_path)

        # 音频切片
        slices_list = await asyncify(slice)(
            origin, Path('resource/sing/slices'), song_id, size_ms=duration * 1000
        )
        if not slices_list or chunk_index >= len(slices_list):
            if chunk_index == len(slices_list):
                await asyncify(splice)(Path("NotExists"), Path('resource/sing/splices'), True, song_id, chunk_index, speaker, key=0)
            logger.error('slice failed', song_id)
            await failed()

        chunk = slices_list[chunk_index]

        # 在显存密集操作前后清理显存
        torch.cuda.empty_cache()

        # 人声分离
        separated = await asyncify(separate)(chunk, Path('resource/sing'), locker=gpu_locker, key=0)  # 不对伴奏变调
        if not separated:
            logger.error('separate failed', song_id)
            await separated_failed()

        vocals, no_vocals = separated

        # 显存清理
        torch.cuda.empty_cache()

        # 音色转换（SVC），对人声进行变调
        svc = await asyncify(inference)(vocals, Path('resource/sing/svc'), speaker=speaker, locker=gpu_locker, key=key)
        if not svc:
            logger.error('svc failed', song_id)
            # 删除 slices 目录下对应的缓存文件
            try:
                for file in Path('resource/sing/slices').glob(f'{song_id}_*.*'):
                    file.unlink()
                    logger.info(f'Deleted slice cache: {file.resolve()}')
            except Exception as e:
                logger.error(f'Failed to delete slice cache: {e}')
                pass

            # 删除 hdemucs_mmi 目录下对应的缓存文件
            try:
                for file in Path('resource/sing/hdemucs_mmi').glob(f'{song_id}_*'):
                    if file.is_dir():
                        # 递归删除目录
                        for sub_file in file.rglob("*"):
                            sub_file.unlink()
                        file.rmdir()
                        logger.info(f"Deleted cache directory: {file.resolve()}")
                    else:
                        # 删除文件
                        file.unlink()
                        logger.info(f"Deleted cache file: {file.resolve()}")
            except Exception as e:
                logger.error(f'Failed to delete hdemucs_mmi cache: {e}')
                pass
            await svc_failed()

        # 显存清理
        torch.cuda.empty_cache()

        # 混合人声和伴奏，伴奏保持原调
        result = await asyncify(mix)(svc, no_vocals, vocals, Path("resource/sing/mix"), svc.stem)
        if not result:
            logger.error('mix failed', song_id)
            await failed()

        # 显存清理
        torch.cuda.empty_cache()

        # 混音后合并混音结果
        if duration > total_duration:
            finished = True
        else:
            finished = chunk_index == len(slices_list) - 1
        await asyncify(splice)(result, Path('resource/sing/splices'), finished, song_id, chunk_index, speaker, key=key)

        # 删除 slices 目录下对应的缓存文件
        try:
            for file in Path('resource/sing/slices').glob(f'{song_id}_*.*'):
                file.unlink()
                logger.info(f'Deleted slice cache: {file.resolve()}')
        except Exception as e:
            logger.error(f'Failed to delete slice cache: {e}')
            pass

        # 删除 hdemucs_mmi 目录下对应的缓存文件
        try:
            for file in Path('resource/sing/hdemucs_mmi').glob(f'{song_id}_*'):
                if file.is_dir():
                    # 递归删除目录
                    for sub_file in file.rglob("*"):
                        sub_file.unlink()
                    file.rmdir()
                    logger.info(f"Deleted cache directory: {file.resolve()}")
                else:
                    # 删除文件
                    file.unlink()
                    logger.info(f"Deleted cache file: {file.resolve()}")
        except Exception as e:
            logger.error(f'Failed to delete hdemucs_mmi cache: {e}')
            pass

        await success(result)

    except FinishedException:
        # 忽略 FinishedException，因为它是正常的流程控制
        pass

    except Exception as e:
        error_message = str(e)
        logger.error(f"An error occurred: {error_message}")
        await failed(error_message)


# 随机放歌（bushi
async def play_song(bot: Bot, event: Event, state: T_State) -> bool:
    text = event.get_plaintext()
    if not text or not text.endswith(SING_CMD):
        return False

    for name, speaker in plugin_config.sing_speakers.items():
        if not text.startswith(name):
            continue
        state['speaker'] = speaker
        return True

    return False


play_cmd = on_message(
    rule=Rule(play_song),
    priority=13,
    block=False,
    permission=permission.GROUP)


SONG_PATH = 'resource/sing/splices/'
MUSIC_PATH = 'resource/music/'


def get_random_song(speaker: str = ""):
    all_song = []
    if os.path.exists(SONG_PATH):
        all_song = [SONG_PATH + s for s in os.listdir(SONG_PATH) \
                    # 只唱过一段的大概率不是什么好听的，排除下
                    if speaker in s and '_spliced0' not in s]
    if not all_song and os.path.exists(LOCAL_MUSIC_PATH):
        all_song = [str(file) for file in Path(LOCAL_MUSIC_PATH).glob("*.mp3")]
    if not all_song:
        all_song = [MUSIC_PATH + s for s in os.listdir(MUSIC_PATH)]

    if not all_song:
        return None
    return random.choice(all_song)


@play_cmd.handle()
async def _(bot: Bot, event: Event, state: T_State):
    config = GroupConfig(event.group_id, cooldown=10)
    if not config.is_cooldown('music'):
        return
    config.refresh_cooldown('music')

    speaker = state['speaker']
    rand_music = get_random_song(speaker)
    if not rand_music:
        return

    if '_spliced' in rand_music:
        splited = Path(rand_music).stem.split('_')
        config.update_sing_progress({
            'song_id': splited[0],
            'chunk_index': int(splited[1].replace('spliced', '')) + 1,
        })
    elif '_full_' in rand_music:
        config.update_sing_progress({
            'song_id': Path(rand_music).stem.split('_')[0],
            'chunk_index': 114514,
        })
    else:
        config.update_sing_progress({
            'song_id': '',
            'chunk_index': 114514,
        })

    with open((rand_music), 'rb') as f:    
      data = f.read()
    msg: Message = MessageSegment.record(file=data)
    await play_cmd.finish(msg)


async def what_song(bot: "Bot", event: "Event", state: T_State) -> bool:
    text = event.get_plaintext()
    return any([text.startswith(spk) for spk in plugin_config.sing_speakers.keys()]) \
        and any(key in text for key in ['什么歌', '哪首歌', '啥歌'])


song_title_cmd = on_message(
    rule=Rule(what_song),
    priority=13,
    block=True,
    permission=permission.GROUP)


@song_title_cmd.handle()
async def _(bot: Bot, event: Event, state: T_State):
    config = GroupConfig(event.group_id, cooldown=10)
    progress = config.sing_progress()
    if not progress:
        return

    if not config.is_cooldown('song_title'):
        return
    config.refresh_cooldown('song_title')

    song_id = progress['song_id']
    song_title = await asyncify(get_song_title)(song_id)
    if not song_title:
        return

    await song_title_cmd.finish(f'{song_title}')


cleanup_sched = require('nonebot_plugin_apscheduler').scheduler


@cleanup_sched.scheduled_job('cron', hour=4, minute=15)
def cleanup_cache():
    logger.info('cleaning up cache...')

    cache_size = plugin_config.song_cache_size
    cache_days = plugin_config.song_cache_days
    current_time = time.time()
    song_atime = {}

    for file_path in Path(SONG_PATH).glob(f"**\*.*"):
        try:
            last_access_time = os.path.getatime(file_path)
        except OSError:
            continue
        song_atime[file_path] = last_access_time
    # 只保留最近最多 cache_size 首歌
    recent_songs = sorted(song_atime, key=song_atime.get, reverse=True)[
        :cache_size]

    prefix_path = 'resource/sing'
    cache_dirs = [Path(prefix_path, suffix) for suffix in [
        'hdemucs_mmi', 'mix', 'ncm', 'slices', 'splices', 'svc']]
    removed_files = 0

    for dir_path in cache_dirs:
        for file_path in dir_path.glob(f"**\*.*"):
            if file_path in recent_songs:
                continue
            try:
                last_access_time = os.path.getatime(file_path)
            except OSError:
                continue
            # 清理超过 cache_days 天未访问的文件
            if (current_time - last_access_time) > (24*60*60) * cache_days:
                os.remove(file_path)
                removed_files += 1

    logger.info(f'cleaned up {removed_files} files.')


def get_song_duration(file_path: Path) -> int:
    """
    获取音频文件的总时长（单位：秒）
    """
    try:
        info = mediainfo(str(file_path))
        duration = float(info['duration'])  # 确保 duration 是浮点数
        if duration > 3600:  # 如果 duration 明显超出合理范围，可能是毫秒单位
            duration = duration / 1000  # 转换为秒
        return int(duration)  # 返回整数秒
    except Exception as e:
        logger.error(f"Failed to get duration for {file_path}: {e}")
        return 0


delete_cache_cmd = on_command(
    "删除缓存",
    priority=10,
    block=True,
    permission=SUPERUSER
)

@delete_cache_cmd.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    text = args.extract_plain_text().strip()
    parts = text.split()
    if len(parts) != 2:
        await delete_cache_cmd.finish("格式错误，请使用：/删除缓存 <speaker> <歌曲名>")

    speaker_name, song_name = parts
    logger.info(f"Deleting cache for speaker: {speaker_name}, song: {song_name}")

    # 从配置文件中获取 speaker
    speaker = plugin_config.sing_speakers.get(speaker_name)
    if not speaker:
        await delete_cache_cmd.finish(f"未找到对应的 speaker：{speaker_name}，请检查输入是否正确。")

    # 优先从 local_music 中获取 song_id
    file_path, song_id = await get_local_song_with_id(song_name)
    if not song_id:
        # 如果 local_music 中未找到，则尝试从 ncm 获取 song_id
        song_id = await asyncify(get_song_id)(song_name)
        if not song_id:
            await delete_cache_cmd.finish(f"未找到歌曲：{song_name}，请检查歌曲名称是否正确。")

    logger.info(f"Resolved song_name '{song_name}' to song_id '{song_id}' for speaker '{speaker}'")

    # 定义缓存目录
    cache_dirs_no_speaker = [
        Path('resource/sing/slices'),
        Path('resource/sing/hdemucs_mmi')
    ]
    cache_dirs_with_speaker = [
        Path('resource/sing/mix'),
        Path('resource/sing/splices'),
        Path('resource/sing/svc')
    ]
    deleted_files = 0

    # 删除不包含 speaker 的缓存（不包括 ncm）
    for cache_dir in cache_dirs_no_speaker:
        logger.info(f"Scanning directory (no speaker): {cache_dir.resolve()}")
        if not cache_dir.exists():
            logger.warning(f"Directory does not exist: {cache_dir.resolve()}")
            continue

        # 匹配文件
        matched_files = list(cache_dir.glob(f"{song_id}_*"))
        logger.info(f"Matched files in {cache_dir.resolve()}: {[str(file) for file in matched_files]}")

        for file in matched_files:
            try:
                if file.is_dir():
                    # 递归删除目录
                    for sub_file in file.rglob("*"):
                        sub_file.unlink()
                    file.rmdir()
                    logger.info(f"Deleted cache directory: {file.resolve()}")
                else:
                    # 删除文件
                    file.unlink()
                    logger.info(f"Deleted cache file: {file.resolve()}")
                deleted_files += 1
            except Exception as e:
                logger.error(f"Failed to delete cache file or directory {file.resolve()}: {e}")

    # 删除包含 speaker 的缓存
    for cache_dir in cache_dirs_with_speaker:
        logger.info(f"Scanning directory (with speaker): {cache_dir.resolve()}")
        if not cache_dir.exists():
            logger.warning(f"Directory does not exist: {cache_dir.resolve()}")
            continue

        # 匹配文件
        matched_files = list(cache_dir.glob(f"{song_id}_*_{speaker}*"))
        logger.info(f"Matched files in {cache_dir.resolve()}: {[str(file) for file in matched_files]}")

        for file in matched_files:
            try:
                if file.is_dir():
                    # 递归删除目录
                    for sub_file in file.rglob("*"):
                        sub_file.unlink()
                    file.rmdir()
                    logger.info(f"Deleted cache directory: {file.resolve()}")
                else:
                    # 删除文件
                    file.unlink()
                    logger.info(f"Deleted cache file: {file.resolve()}")
                deleted_files += 1
            except Exception as e:
                logger.error(f"Failed to delete cache file or directory {file.resolve()}: {e}")

    if deleted_files > 0:
        await delete_cache_cmd.finish(f"已删除 {deleted_files} 个缓存文件。")
    else:
        await delete_cache_cmd.finish("未找到相关缓存文件。")

menu_cmd = on_command(
    cmd="唱歌菜单", 
    permission=permission.GROUP
    #rule=to_me()
)

@menu_cmd.handle()
async def handle_menu(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """
    发送唱歌菜单
    """
    text = args.extract_plain_text().strip()
    if text == "完整版":
        # 发送完整版菜单
        menu_msg = f"""
【唱歌菜单 - 完整版】
━━━━━━━━━━━━━━
基础命令：<speaker>唱歌 <歌曲名>
可选参数：
    -t <时长>：指定时长（秒），默认 {plugin_config.sing_length} 秒
    -k <变调>：指定音高变化，单位key，
       范围 -12 到 12，默认 0
    -s <歌曲源>：指定歌曲源，可以填入<ncm>或<local>，
       若填入<ncm>，则忽略本地歌曲库，
若填入<local>，则优先查询本地曲库，
       若本地曲库中没有则回退到一般流程，
       若不指定，默认为<local>.
示例：牛牛唱歌 富士山下 -t 300 -k 12 -s ncm
建议：女声模型唱男声歌曲时升调12个key，反之则降低12个key
注：有时传入-t参数时可能会没有效果，这是因为<speaker>之前已经唱过这首歌了，所以会调用缓存中的成曲。
此时再输入一次相同的命令即可删除缓存，按照新的时间推理。

关于报错：牛牛唱歌出错有两种情况，
1.报错信息的出现时间距离"欢呼吧！"比较近，这种情况一般是因为删除缓存时出错或者单位时间内请求太多次导致被网易云暂时封禁掉，请等一会再重试。
2.报错信息出现的时间距离"欢呼吧！"比较远，这种情况一般是因为 torch.OutOfMemory ，也就是输入的歌曲长度为牛牛的8G显存的P104的不可承受之重，此时，若你传入了 -t 参数则请删去此参数，若你没传入 -t 参数则请传入一个较小的-t参数，比如 -t 60 .

清除缓存：清除缓存 <speaker> <歌曲名>（本命令权限级别为SUPERUSER）

牛牛点歌：牛牛点歌 <歌曲名>，此功能有30秒全局冷却
跳过牛牛点歌冷却：跳过牛牛点歌冷却（本命令权限级别为SUPERUSER）

本地曲库管理：
1.列出本地曲库
可选参数：
    -p <页码>：指定页码，默认第1页，范围 N* .
示例：列出本地曲库 -p 2
2.刷新本地曲库

发送歌曲文件：发送歌曲文件 <speaker> <歌曲名>
示例：发送歌曲文件 牛牛 富士山下
注1：本命令每人每24小时有5次调用限制，每群每24小时有30次调用限制，这两者是“或”的关系
注2：该命令的实现方式为上传至群文件，若牛牛没有群文件上传权限则无法使用。
━━━━━━━━━━━━━━
当前的可选speaker：{', '.join(plugin_config.sing_speakers.keys())}
━━━━━━━━━━━━━━
soyo唱不了春日影不是bug哦，是小彩蛋，要是想让soyo唱的话可以在唱歌命令最后加上"--soyo-force"
        """.strip()
    else:
        # 发送简略版菜单
        menu_msg = f"""
【唱歌菜单 - 简略版】
━━━━━━━━━━━━━━ 
基础命令：<speaker>唱歌 <歌曲名>
示例：牛牛唱歌 富士山下
牛牛点歌：牛牛点歌 <歌曲名>
发送歌曲文件：发送歌曲文件 <speaker> <歌曲名>
本地曲库管理：列出本地曲库
━━━━━━━━━━━━━━
当前的可选speaker：{', '.join(plugin_config.sing_speakers.keys())}
━━━━━━━━━━━━━━
输入“唱歌菜单 完整版”查看详细命令说明。
        """.strip()

    await bot.send(event, menu_msg)

request_song_msg = on_message(
    rule=Rule(lambda bot, event, state: event.get_plaintext().startswith("牛牛点歌")),
    priority=10,
    block=True,
    permission=permission.GROUP
)

# 全局冷却时间
global_request_song_cooldown = 0

@request_song_msg.handle()
async def handle_request_song(bot: Bot, event: GroupMessageEvent, matcher: Matcher):
    global global_request_song_cooldown
    current_time = time.time()

    # 检查是否为超级用户
    if str(event.user_id) in bot.config.superusers:
        logger.info(f"超级用户 {event.user_id} 跳过全局点歌冷却")
    else:
        # 检查全局冷却时间
        if current_time - global_request_song_cooldown < 30:  # 全局冷却时间为 30 秒
            await matcher.finish("点歌冷却中，请稍后再试喵！")

        # 更新全局冷却时间
        global_request_song_cooldown = current_time

    text = event.get_plaintext().strip()
    if text == "牛牛点歌":
        # 从缓存中随机选取一首歌
        cached_songs = list(Path("resource/sing/ncm").glob("*.mp3"))
        if not cached_songs:
            await request_song_msg.finish("缓存中没有可用的歌曲喵！")
            return

        random_song = random.choice(cached_songs)
        try:
            with open(random_song, 'rb') as f:
                data = f.read()
            msg: Message = MessageSegment.record(file=data)
            await request_song_msg.finish(msg)
        except FinishedException:
        # 忽略 FinishedException，因为它是正常的流程控制
            pass
        except Exception as e:
            logger.error(f"发送随机歌曲失败：{e}")
            await request_song_msg.finish("发送随机歌曲失败，请稍后重试喵！")
        return

    if len(text.split()) < 2:
        await request_song_msg.finish("格式错误，请使用：牛牛点歌 <歌曲名>")
        return

    song_name = text.replace("牛牛点歌", "").strip()

    # 优先从本地歌曲库中查找歌曲
    local_song = await get_local_song(song_name)
    if local_song:
        try:
            with open(local_song, 'rb') as f:
                data = f.read()
            msg: Message = MessageSegment.record(file=data)
            await request_song_msg.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送本地歌曲失败：{e}")
            await request_song_msg.finish("发送本地歌曲失败，请稍后重试喵！")
        return

    song_id = await asyncify(get_song_id)(song_name)
    if not song_id:
        await request_song_msg.finish(f"未找到歌曲：{song_name}，请检查歌曲名称是否正确。")
        return

    song_path = Path(f'resource/sing/ncm/{song_id}.mp3')
    if not song_path.exists():
        # 若缓存不存在，则下载歌曲
        await request_song_msg.send(f"未找到缓存，正在下载歌曲：{song_name}...")
        song_path = await asyncify(download)(song_id)
        if not song_path:
            await request_song_msg.finish(f"下载失败：{song_name}，请稍后重试。")
            return

    try:
        with open(song_path, 'rb') as f:
            data = f.read()
        msg: Message = MessageSegment.record(file=data)
        await request_song_msg.finish(msg)
    except FinishedException:
        # 忽略 FinishedException，因为它是正常的流程控制
        pass
    except Exception as e:
        logger.error(f"发送歌曲失败：{e}")
        await request_song_msg.finish("发送歌曲失败，请稍后重试。")

skip_cooldown_cmd = on_command(
    "跳过牛牛点歌冷却",
    priority=10,
    block=True,
    permission=SUPERUSER
)

@skip_cooldown_cmd.handle()
async def handle_skip_cooldown(bot: Bot, event: GroupMessageEvent):
    global global_request_song_cooldown
    if str(event.user_id) not in bot.config.superusers:  # 修复权限检查逻辑
        await skip_cooldown_cmd.finish("只有超级用户可以使用此命令喵！")
        return

    global_request_song_cooldown = 0  # 重置全局冷却时间
    await skip_cooldown_cmd.finish("已跳过全局牛牛点歌冷却时间。")

list_local_songs_cmd = on_command(
    "列出本地曲库",
    priority=10,
    block=True,
    permission=permission.GROUP
)

@list_local_songs_cmd.handle()
async def handle_list_local_songs(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """
    分页列出本地曲库中的所有歌曲，按 local_song_id 排序，并在歌曲名前添加序号
    """
    if not os.path.exists(LOCAL_MUSIC_PATH):
        await list_local_songs_cmd.finish("本地曲库不存在喵！")
    sorted_songs = sorted(local_music_ids.items(), key=lambda item: int(item[1]))
    songs = [(song_id, Path(file_path).stem) for file_path, song_id in sorted_songs]

    if not songs:
        await list_local_songs_cmd.finish("本地曲库中没有歌曲喵！")

    # 默认页码为 1
    page = 1
    try:
        text = args.extract_plain_text().strip()
        if "-p " in text:
            p_pos = text.find("-p ")
            page = int(text[p_pos + 3:].split()[0])  # 获取 -p 后面的页码
            if page <= 0:
                raise ValueError
    except ValueError:
        await list_local_songs_cmd.finish("页码参数无效喵，请输入一个正整数！")

    # 每页显示 10 条
    page_size = 10
    total_pages = (len(songs) + page_size - 1) // page_size

    if page > total_pages:
        await list_local_songs_cmd.finish(f"页码超出范围喵！当前共有 {total_pages} 页。")

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    songs_on_page = songs[start_index:end_index]

    # 添加序号
    song_list = "\n".join([f"{start_index + i + 1}. {song_name}" for i, (song_id, song_name) in enumerate(songs_on_page)])
    await list_local_songs_cmd.finish(
        f"本地曲库中的歌曲如下喵（第 {page}/{total_pages} 页）：\n{song_list}"
    )

send_song_file_cmd = on_command(
    "发送歌曲文件",
    priority=10,
    block=True,
    permission=permission.GROUP
)

from shutil import copy2  # 添加导入  # 添加导入
from collections import defaultdict  # 添加导入
from datetime import datetime, timedelta  # 添加导入
import json  # 添加导入

CACHE_DIR = Path("data/sing/cache").resolve()  # 定义缓存目录为机器人根目录的 data 目录
CACHE_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

COOLDOWN_FILE = Path("data/sing/cooldowns.json")  # 冷却时间记录文件

# 定义冷却时间限制
USER_COOLDOWN_LIMIT = 5  # 每人每天限制 5 次
GROUP_COOLDOWN_LIMIT = 30  # 每群每天限制 50 次

# 加载冷却时间记录
if COOLDOWN_FILE.exists():
    with open(COOLDOWN_FILE, "r", encoding="utf-8") as f:
        cooldown_data = json.load(f)
    user_cooldowns = defaultdict(list, {int(k): [datetime.fromisoformat(ts) for ts in v] for k, v in cooldown_data.get("user", {}).items()})
    group_cooldowns = defaultdict(list, {int(k): [datetime.fromisoformat(ts) for ts in v] for k, v in cooldown_data.get("group", {}).items()})
else:
    user_cooldowns = defaultdict(list)
    group_cooldowns = defaultdict(list)

def save_cooldowns():
    """将冷却时间记录持久化到文件"""
    with open(COOLDOWN_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "user": {k: [ts.isoformat() for ts in v] for k, v in user_cooldowns.items()},
            "group": {k: [ts.isoformat() for ts in v] for k, v in group_cooldowns.items()}
        }, f, ensure_ascii=False, indent=4)

LOCAL_MUSIC_ID_FILE = Path("data/sing/local_music_ids.json")  # 定义本地歌曲 ID 映射文件

# 加载或初始化本地歌曲 ID 映射
if LOCAL_MUSIC_ID_FILE.exists():
    with open(LOCAL_MUSIC_ID_FILE, "r", encoding="utf-8") as f:
        local_music_ids = json.load(f)
else:
    local_music_ids = {}

def save_local_music_ids():
    """将本地歌曲 ID 映射持久化到文件"""
    with open(LOCAL_MUSIC_ID_FILE, "w", encoding="utf-8") as f:
        json.dump(local_music_ids, f, ensure_ascii=False, indent=4)

def assign_local_music_ids():
    """
    为本地歌曲分配 song_id，仅为新歌曲分配 ID，不覆盖已有记录
    """
    global local_music_ids
    local_music_files = sorted(
        Path(LOCAL_MUSIC_PATH).glob("*.mp3"),
        key=lambda f: f.stat().st_mtime  # 按修改时间排序
    )
    next_id = max(
        (int(song_id) for song_id in local_music_ids.values() if song_id.isdigit()),
        default=0
    ) + 1  # 确保新分配的 ID 不与现有 ID 冲突

    for file in local_music_files:
        if str(file) not in local_music_ids:
            local_music_ids[str(file)] = f"{next_id:04d}"  # 分配格式为 00nnnn
            next_id += 1

    save_local_music_ids()  # 保存更新后的 local_music_ids.json

assign_local_music_ids()  # 初始化时分配 ID

async def get_local_song_with_id(song_name: str) -> tuple[Path, str]:
    """
    从本地歌曲库中查找歌曲文件，并返回文件路径和对应的 song_id
    """
    for file_path, song_id in local_music_ids.items():
        if song_name.lower() in Path(file_path).stem.lower():
            return Path(file_path), song_id
    return None, None

EXEMPT_USERS_FILE = Path("data/sing/exempt_users.json")  # 定义豁免列表文件路径
EXEMPT_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

# 加载豁免用户列表
if EXEMPT_USERS_FILE.exists():
    with open(EXEMPT_USERS_FILE, "r", encoding="utf-8") as f:
        exempt_users = set(json.load(f))
else:
    exempt_users = set()

def save_exempt_users():
    """保存豁免用户列表到文件"""
    with open(EXEMPT_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(exempt_users), f, ensure_ascii=False, indent=4)

exempt_users_cmd = on_command(
    "发送歌曲文件豁免",
    priority=10,
    block=True,
    permission=SUPERUSER
)

@exempt_users_cmd.handle()
async def handle_exempt_users(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """
    添加或删除豁免用户
    """
    text = args.extract_plain_text().strip()
    parts = text.split()
    if len(parts) != 2 or parts[0] not in ["添加", "删除"]:
        await exempt_users_cmd.finish("格式错误，请使用：发送歌曲文件豁免 添加/删除 user_id")

    action, user_id = parts
    if not user_id.isdigit():
        await exempt_users_cmd.finish("user_id 必须是数字！")

    user_id = int(user_id)
    if action == "添加":
        if user_id in exempt_users:
            await exempt_users_cmd.finish(f"用户 {user_id} 已在豁免列表中！")
        exempt_users.add(user_id)
        save_exempt_users()
        await exempt_users_cmd.finish(f"已将用户 {user_id} 添加到豁免列表！")
    elif action == "删除":
        if user_id not in exempt_users:
            await exempt_users_cmd.finish(f"用户 {user_id} 不在豁免列表中！")
        exempt_users.remove(user_id)
        save_exempt_users()
        await exempt_users_cmd.finish(f"已将用户 {user_id} 从豁免列表中删除！")

@send_song_file_cmd.handle()
async def handle_send_song_file(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """
    发送 splices 文件夹中的推理好的歌曲文件
    """
    user_id = event.user_id
    group_id = event.group_id
    now = datetime.now()

    # 检查是否为豁免用户
    if user_id in exempt_users:
        logger.info(f"用户 {user_id} 在豁免列表中，跳过限制检查")
    elif str(user_id) not in bot.config.superusers:
        # 检查用户冷却时间
        user_cooldowns[user_id] = [time for time in user_cooldowns[user_id] if now - time < timedelta(days=1)]
        if len(user_cooldowns[user_id]) >= USER_COOLDOWN_LIMIT:
            await send_song_file_cmd.finish("你今天已经发送了 5 次歌曲文件喵，请明天再试！")

        # 检查群冷却时间
        group_cooldowns[group_id] = [time for time in group_cooldowns[group_id] if now - time < timedelta(days=1)]
        if len(group_cooldowns[group_id]) >= GROUP_COOLDOWN_LIMIT:
            await send_song_file_cmd.finish("本群今天已经发送了 30 次歌曲文件喵，请明天再试！")

        # 记录调用时间
        user_cooldowns[user_id].append(now)
        group_cooldowns[group_id].append(now)
        save_cooldowns()  # 保存冷却时间记录

    text = args.extract_plain_text().strip()
    logger.info(f"收到的命令参数: {text}")  # 记录输入参数

    parts = text.split()
    if len(parts) < 2:
        await send_song_file_cmd.finish("格式错误，请使用：发送歌曲文件 <speaker> <歌曲名称> [-k <key值>]")

    speaker_name = parts[0]
    key_val = 0  # 默认 key 值为 0

    # 解析 -k 参数
    if "-k" in parts:
        try:
            k_index = parts.index("-k")
            key_val = int(parts[k_index + 1])  # 获取 -k 后面的值
            if key_val < -12 or key_val > 12:
                raise ValueError
            parts = parts[:k_index]  # 移除 -k 参数及其值
        except (ValueError, IndexError):
            await send_song_file_cmd.finish("key 参数无效喵，请输入范围在 -12 到 12 的整数！")

    song_name = " ".join(parts[1:])  # 剩余部分作为歌曲名称

    logger.info(f"解析出的 speaker: {speaker_name}, song_name: {song_name}")  # 记录解析结果
    logger.info(f"解析出的 key 值: {key_val}")  # 记录 key 值

    # 获取处理后的 speaker 名称
    speaker = plugin_config.sing_speakers.get(speaker_name)
    if not speaker:
        await send_song_file_cmd.finish(f"未找到对应的 speaker：{speaker_name}，请检查输入是否正确喵！")
    logger.info(f"处理后的 speaker 名称: {speaker}")  # 记录处理后的 speaker 名称

    # 优先从 local_music 中获取 song_id
    file_path, song_id = await get_local_song_with_id(song_name)
    if file_path:
        # 构造 splices 文件路径
        file_pattern = f"{song_id}_*_{key_val}key_{speaker}.mp3"
        splices_file_path = next(Path("resource/sing/splices").glob(file_pattern), None)
        logger.debug(f"尝试匹配 splices 文件路径: {file_pattern}")  # 添加调试日志
        if not splices_file_path:
            # 查找符合 speaker 和歌曲名的所有 key 值
            available_keys = [
                int(file.stem.split('_')[-2].replace('key', ''))
                for file in Path("resource/sing/splices").glob(f"{song_id}_*_{speaker}.mp3")
            ]
            if available_keys:
                available_keys_str = ", ".join(map(str, sorted(available_keys)))
                await send_song_file_cmd.finish(
                    f"指定的key值不存在喵！当前可用的key值有：{available_keys_str}"
                )
            else:
                logger.warning(f"未找到 splices 文件，使用 local_music 原曲：{file_path}")
        else:
            logger.info(f"匹配到的 splices 文件路径: {splices_file_path}")  # 记录匹配到的文件路径
    else:
        # 如果 local_music 中未找到，则尝试从 ncm 获取 song_id
        song_id = await asyncify(get_song_id)(song_name)
        if song_id:
            # 构造 ncm 文件路径
            file_pattern = f"{song_id}_*_{key_val}key_{speaker}.mp3"
            splices_file_path = next(Path("resource/sing/splices").glob(file_pattern), None)
            logger.debug(f"尝试匹配 ncm 文件路径: {file_pattern}")  # 添加调试日志
            if not splices_file_path:
                # 查找符合 speaker 和歌曲名的所有 key 值
                available_keys = [
                    int(file.stem.split('_')[-2].replace('key', ''))
                    for file in Path("resource/sing/splices").glob(f"{song_id}_*_{speaker}.mp3")
                ]
                if available_keys:
                    available_keys_str = ", ".join(map(str, sorted(available_keys)))
                    await send_song_file_cmd.finish(
                        f"指定的key值不存在喵！当前可用的key值有：{available_keys_str}"
                    )
            else:
                logger.info(f"匹配到的 ncm 文件路径: {splices_file_path}")  # 记录匹配到的文件路径

    # 检查最终匹配到的文件路径
    if not splices_file_path or not splices_file_path.exists():
        logger.error(f"未找到歌曲文件：{song_name}（{key_val} key），file_path: {file_path}, splices_file_path: {splices_file_path}")
        await send_song_file_cmd.finish(f"未找到歌曲文件：{song_name}（{key_val} key）喵！")

    try:
        # 将文件复制到缓存目录
        cached_file_path = CACHE_DIR / splices_file_path.name  # 修复为使用 splices_file_path
        copy2(splices_file_path, cached_file_path)  # 修复为复制 splices_file_path
        logger.info(f"文件已复制到缓存目录: {cached_file_path}")

        # 重命名文件为 "<speaker> 歌曲名 key值"
        new_file_name = f"{speaker_name} {song_name} {key_val}key.mp3"
        renamed_file_path = CACHE_DIR / new_file_name
        cached_file_path.rename(renamed_file_path)
        logger.info(f"文件已重命名为: {renamed_file_path}")

        # 检查文件是否存在于缓存目录
        if not renamed_file_path.exists():
            logger.error(f"缓存目录中未找到文件: {renamed_file_path}")
            await send_song_file_cmd.finish("缓存文件不存在，发送失败喵！")

        # 使用 upload_file 方法发送文件
        await upload_file(bot, event, renamed_file_path.name, path=str(renamed_file_path))
        await send_song_file_cmd.finish(f"已发送歌曲文件：{renamed_file_path.name} 喵！")
    except FinishedException:
        # 忽略 FinishedException，因为它是正常的流程控制
        pass
    except Exception as e:
        logger.error(f"发送歌曲文件失败：{e}")
        await send_song_file_cmd.finish("发送歌曲文件失败，请稍后重试喵！")

refresh_local_music_cmd = on_command(
    "刷新本地曲库",
    priority=10,
    block=True,
    permission=SUPERUSER
)

@refresh_local_music_cmd.handle()
async def handle_refresh_local_music(bot: Bot, event: GroupMessageEvent):
    """
    刷新本地曲库，将新加入 local_music 的歌曲写入 local_music_ids.json
    """
    try:
        # 调用 assign_local_music_ids 重新分配本地歌曲 ID
        assign_local_music_ids()
        await refresh_local_music_cmd.finish("本地曲库已刷新，新增歌曲已写入 local_music_ids.json 喵！")
    except FinishedException:
        # 忽略 FinishedException，因为它是正常的流程控制
        pass
    except Exception as e:
        logger.error(f"刷新本地曲库失败：{e}")
        await refresh_local_music_cmd.finish("刷新本地曲库失败，请稍后重试喵！")