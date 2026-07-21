import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SafeBirthEnv(gym.Env):
    """
    SAFEBIRTH Maternal Health Triage Environment
    A Community Health Worker agent triages pregnant women
    showing danger signs and decides appropriate care level.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = (900, 600)

        # 11 continuous observations: vitals + danger signs + context
        self.observation_space = spaces.Box(
            low=np.array([60,40,40,35,8,0,0,0,20,0,0], dtype=np.float32),
            high=np.array([200,130,180,42,40,1,1,1,42,100,10], dtype=np.float32),
            dtype=np.float32
        )

        # 5 discrete triage actions
        self.action_space = spaces.Discrete(5)

        self.action_names = [
            "Monitor at Home",
            "Refer to Health Post",
            "Refer to Clinic",
            "Emergency Hospital Referral",
            "Call Ambulance"
        ]

        self.current_step = 0
        self.max_steps = 10
        self.state = None
        self.true_severity = None
        self.episode_reward = 0
        self.last_action = None
        self.last_reward = 0
        self.outcome_message = ""

    def _get_severity(self, obs):
        systolic, diastolic, hr, temp, rr, bleeding, convulsions, loc, ga, distance, _ = obs
        score = 0
        if systolic >= 160 or diastolic >= 110:
            score += 3
        elif systolic >= 140 or diastolic >= 90:
            score += 2
        if hr > 120 or hr < 50:
            score += 2
        elif hr > 100:
            score += 1
        if temp > 39.5 or temp < 36:
            score += 1
        if bleeding:
            score += 3
        if convulsions:
            score += 4
        if loc:
            score += 4
        if score >= 8:
            return 4
        elif score >= 5:
            return 3
        elif score >= 3:
            return 2
        elif score >= 1:
            return 1
        else:
            return 0

    def _compute_reward(self, action, true_severity):
        difference = abs(action - true_severity)
        if difference == 0:
            return 20, "Perfect triage decision!"
        elif difference == 1:
            if action < true_severity:
                return -10, "Slight under-triage - patient needs more care"
            else:
                return -3, "Slight over-triage - unnecessary referral"
        elif difference == 2:
            if action < true_severity:
                return -25, "Under-triage - patient condition may worsen"
            else:
                return -8, "Over-triage - straining health resources"
        else:
            if action < true_severity:
                return -50, "CRITICAL under-triage - patient at risk!"
            else:
                return -15, "Severe over-triage"

    def _generate_patient(self):
        profile = self.np_random.integers(0, 5)
        if profile == 0:
            systolic = self.np_random.uniform(100, 130)
            diastolic = self.np_random.uniform(60, 85)
            hr = self.np_random.uniform(60, 90)
            temp = self.np_random.uniform(36.5, 37.5)
            rr = self.np_random.uniform(12, 20)
            bleeding, convulsions, loc = 0, 0, 0
        elif profile == 1:
            systolic = self.np_random.uniform(130, 150)
            diastolic = self.np_random.uniform(85, 100)
            hr = self.np_random.uniform(70, 100)
            temp = self.np_random.uniform(36.5, 38.0)
            rr = self.np_random.uniform(14, 22)
            bleeding, convulsions, loc = 0, 0, 0
        elif profile == 2:
            systolic = self.np_random.uniform(150, 180)
            diastolic = self.np_random.uniform(100, 120)
            hr = self.np_random.uniform(80, 120)
            temp = self.np_random.uniform(37.0, 39.0)
            rr = self.np_random.uniform(16, 28)
            bleeding = int(self.np_random.random() > 0.7)
            convulsions = int(self.np_random.random() > 0.8)
            loc = 0
        elif profile == 3:
            systolic = self.np_random.uniform(80, 130)
            diastolic = self.np_random.uniform(40, 80)
            hr = self.np_random.uniform(100, 150)
            temp = self.np_random.uniform(36.0, 38.0)
            rr = self.np_random.uniform(18, 35)
            bleeding = 1
            convulsions = 0
            loc = int(self.np_random.random() > 0.7)
        else:
            systolic = self.np_random.uniform(160, 200)
            diastolic = self.np_random.uniform(110, 130)
            hr = self.np_random.uniform(100, 160)
            temp = self.np_random.uniform(37.5, 42.0)
            rr = self.np_random.uniform(20, 40)
            bleeding = int(self.np_random.random() > 0.5)
            convulsions = 1
            loc = int(self.np_random.random() > 0.5)

        ga = self.np_random.uniform(20, 42)
        distance = self.np_random.uniform(0, 100)
        return np.array([systolic, diastolic, hr, temp, rr,
                         bleeding, convulsions, loc,
                         ga, distance, 0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0
        self.last_action = None
        self.last_reward = 0
        self.outcome_message = ""
        self.state = self._generate_patient()
        self.true_severity = self._get_severity(self.state)
        self.state[10] = float(self.true_severity * 2)
        if self.render_mode == "human":
            self._render_frame()
        return self.state.copy(), {}

    def step(self, action):
        self.current_step += 1
        reward, message = self._compute_reward(int(action), self.true_severity)
        self.episode_reward += reward
        self.last_action = int(action)
        self.last_reward = reward
        self.outcome_message = message
        terminated = self.current_step >= self.max_steps
        if not terminated:
            self.state = self._generate_patient()
            self.true_severity = self._get_severity(self.state)
            self.state[10] = float(self.true_severity * 2)
        info = {
            "true_severity": self.true_severity,
            "action_taken": self.action_names[int(action)],
            "episode_reward": self.episode_reward
        }
        if self.render_mode == "human":
            self._render_frame()
        return self.state.copy(), reward, terminated, False, info

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()

    def _render_frame(self):
        import pygame
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("SAFEBIRTH - Maternal Health Triage RL")
            self.clock = pygame.time.Clock()

        WHITE=(255,255,255); BLACK=(0,0,0); RED=(220,50,50)
        GREEN=(50,180,50); BLUE=(50,100,220); ORANGE=(255,140,0)
        PURPLE=(128,0,128); LIGHT_GRAY=(240,240,240); DARK_GRAY=(100,100,100)
        TEAL=(0,150,136); PINK=(233,30,99)
        severity_colors=[GREEN,(100,200,100),ORANGE,RED,PURPLE]
        severity_labels=["Low","Mild","Moderate","High","Critical"]

        canvas = pygame.Surface(self.window_size)
        canvas.fill(WHITE)

        title_font = pygame.font.SysFont("Arial", 22, bold=True)
        header_font = pygame.font.SysFont("Arial", 16, bold=True)
        body_font = pygame.font.SysFont("Arial", 14)
        small_font = pygame.font.SysFont("Arial", 12)

        pygame.draw.rect(canvas, TEAL, (0, 0, 900, 60))
        canvas.blit(title_font.render("SAFEBIRTH — Maternal Health Triage RL Agent", True, WHITE), (20, 18))
        canvas.blit(header_font.render(f"Step: {self.current_step}/{self.max_steps}", True, WHITE), (750, 22))

        if self.state is not None:
            systolic,diastolic,hr,temp,rr,bleeding,convulsions,loc,ga,distance,_ = self.state

            # Left panel
            pygame.draw.rect(canvas, LIGHT_GRAY, (10,70,280,470), border_radius=8)
            pygame.draw.rect(canvas, TEAL, (10,70,280,35), border_radius=8)
            canvas.blit(header_font.render("Patient Vital Signs", True, WHITE), (20,78))

            vitals = [
                ("Blood Pressure", f"{systolic:.0f}/{diastolic:.0f} mmHg",
                 RED if systolic>=160 else ORANGE if systolic>=140 else GREEN),
                ("Heart Rate", f"{hr:.0f} bpm",
                 RED if hr>120 or hr<50 else ORANGE if hr>100 else GREEN),
                ("Temperature", f"{temp:.1f}C",
                 RED if temp>39.5 else ORANGE if temp>38.5 else GREEN),
                ("Resp. Rate", f"{rr:.0f}/min",
                 RED if rr>30 else ORANGE if rr>25 else GREEN),
                ("Gestational Age", f"{ga:.0f} weeks", BLUE),
                ("Distance", f"{distance:.1f} km",
                 RED if distance>50 else ORANGE if distance>20 else GREEN),
            ]
            y=115
            for label,value,color in vitals:
                canvas.blit(body_font.render(label+":", True, DARK_GRAY), (20,y))
                canvas.blit(body_font.render(value, True, color), (180,y))
                y+=30

            y+=10
            canvas.blit(header_font.render("Danger Signs", True, BLACK), (20,y)); y+=25
            for sign,present in [("Severe Bleeding",bool(bleeding)),("Convulsions",bool(convulsions)),("Loss of Consciousness",bool(loc))]:
                color=RED if present else GREEN
                canvas.blit(body_font.render(f"{sign}: {'YES' if present else 'No'}", True, color),(20,y)); y+=25

            # Middle panel
            pygame.draw.rect(canvas, LIGHT_GRAY, (300,70,280,470), border_radius=8)
            pygame.draw.rect(canvas, PINK, (300,70,280,35), border_radius=8)
            canvas.blit(header_font.render("Triage Assessment", True, WHITE), (310,78))

            sev=self.true_severity if self.true_severity is not None else 0
            sev_color=severity_colors[min(sev,4)]
            canvas.blit(header_font.render("True Severity Level:", True, BLACK),(310,115))
            pygame.draw.rect(canvas, sev_color, (310,140,240,50), border_radius=8)
            canvas.blit(title_font.render(f"Level {sev} — {severity_labels[min(sev,4)]}", True, WHITE),(320,155))

            for i in range(5):
                bar_color=severity_colors[i] if i<=sev else LIGHT_GRAY
                pygame.draw.rect(canvas, bar_color, (310+i*48,205,44,20), border_radius=4)
                canvas.blit(small_font.render(str(i),True,WHITE if i<=sev else DARK_GRAY),(330+i*48,208))

            canvas.blit(header_font.render(f"Correct Action: {self.action_names[sev]}", True, DARK_GRAY),(310,235))

            if self.last_action is not None:
                canvas.blit(header_font.render("Agent Decision:", True, BLACK),(310,270))
                action_color=GREEN if self.last_action==self.true_severity else RED
                canvas.blit(body_font.render(f"{self.action_names[self.last_action]}", True, action_color),(310,293))
                canvas.blit(body_font.render(f"Reward: {self.last_reward:+.0f}", True, GREEN if self.last_reward>0 else RED),(310,316))
                canvas.blit(small_font.render(self.outcome_message[:45], True, DARK_GRAY),(310,345))

            # Right panel
            pygame.draw.rect(canvas, LIGHT_GRAY, (590,70,300,470), border_radius=8)
            pygame.draw.rect(canvas, PURPLE, (590,70,300,35), border_radius=8)
            canvas.blit(header_font.render("Episode Statistics", True, WHITE),(600,78))

            y2=120
            for stat_label,stat_val in [("Episode Reward",f"{self.episode_reward:.1f}"),("Step",f"{self.current_step}/{self.max_steps}"),("Last Reward",f"{self.last_reward:+.1f}")]:
                canvas.blit(header_font.render(stat_label+":", True, DARK_GRAY),(600,y2))
                val_color=GREEN if "Reward" in stat_label and self.episode_reward>=0 else BLUE
                canvas.blit(title_font.render(stat_val, True, val_color),(600,y2+22))
                y2+=65

            canvas.blit(header_font.render("Action Space:", True, BLACK),(600,y2+10))
            for i,name in enumerate(self.action_names):
                color=GREEN if i==self.true_severity else DARK_GRAY
                canvas.blit(small_font.render(f"{i}: {name}", True, color),(600,y2+35+i*22))

        pygame.draw.rect(canvas, TEAL, (0,550,900,50))
        canvas.blit(small_font.render("SAFEBIRTH RL — Maternal Health AI | African Leadership University", True, WHITE),(20,565))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None