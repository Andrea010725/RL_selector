# save as test_pygame.py
import pygame, os, time
pygame.init()
print("DISPLAY =", os.getenv("DISPLAY"))
print("SDL driver =", pygame.display.get_driver())  # 期望输出 x11
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("pygame x11 ok")
screen.fill((40, 40, 40))
pygame.display.flip()
time.sleep(1.5)
