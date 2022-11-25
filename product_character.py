import codecs
import os
import pygame
chinese_dir = 'chinese_word'
# if not os.path.exists(chinese_dir):
#     os.mkdir(chinese_dir)
def product(word,size):
    pygame.init()
    font_path = "./fonts/wqy-microhei.ttc"
    font = pygame.font.Font(font_path, size)  # 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
    rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
    # print(os.path.join(chinese_dir, word + ".png"))
    return os.path.join(chinese_dir, word + ".png")
# product("12312",100)