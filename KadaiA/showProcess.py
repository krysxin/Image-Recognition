import sys, time

class ShowProcess():
    """処理進度を表すクラス"""
    i = 0 # 現在の進度
    max_steps = 0 # 合計処理数
    max_arrow = 50 #プロセスバーの長さ

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    # 現在の進度iによって処理進度を表す
    # 仕様：[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]m/m
    def show_process(self, i=0):
        if i != 0:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #'>'の表示個数を表す
        num_line = self.max_arrow - num_arrow #'-'の表示個数を表す
        percent = self.i * 100.0 / self.max_steps #完成したプロセス，xx.xx%
        p = self.i #完成した処理回数
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.0f' % p + '/'+ str(self.max_steps) + '\r' 
        sys.stdout.write(process_bar) 
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        self.i = 0

if __name__=='__main__':
    max_steps = 100

    process_bar = ShowProcess(max_steps)

    for i in range(max_steps):
        process_bar.show_process()
        time.sleep(0.01)