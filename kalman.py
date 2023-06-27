class Kalman:
    def __init__(self, x0, P0, R, Q, maxv=-1):
        self.m_x0 = x0
        self.m_P0 = P0
        self.m_R0 = R
        self.m_Q0 = Q
        self.m_x = self.m_x0
        self.m_P = self.m_P0
        self.m_R = self.m_R0
        self.m_Q = self.m_Q0
        self.m_maxv = maxv if maxv != -1 else x0 * 2

    def reset(self):
        self.m_x = self.m_x0
        self.m_P = self.m_P0
        self.m_R = self.m_R0
        self.m_Q = self.m_Q0

    def filter(self, z):
        if z > self.m_maxv:
            z = self.m_maxv

        K = self.m_P / (self.m_P + self.m_R)
        self.m_x = self.m_x + K * (z - self.m_x)
        self.m_P = (1 - K) * self.m_P + self.m_Q

        return self.m_x
