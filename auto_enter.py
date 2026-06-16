import pyautogui
import time

def auto_press_enter():
    print("腳本已啟動。請切換到你要按 Enter 的目標視窗。")
    print("按下 Ctrl+C 可停止執行。")
    
    # 給使用者 3 秒鐘的時間切換視窗
    print("3 秒後開始執行...")
    time.sleep(3)
    
    try:
        while True:
            # 執行按鍵動作
            pyautogui.press('enter')
            print("已按下 Enter 鍵。")
            
            # 等待 10 秒
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n腳本已由使用者停止。")

if __name__ == "__main__":
    auto_press_enter()