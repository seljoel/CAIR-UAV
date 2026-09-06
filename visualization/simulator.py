"""
Pygame Simulator Visualization
===============================
Provides a live, interactive 2D visualization of the CAIR-UAV environment.

Features:
- Renders the 20x20 disaster grid.
- Overlays base station, UAV position, and battery gauge.
- Shows hazard zones (red) and survivor belief (green).
- Displays found survivors.
- Supports pause (SPACE), restart (R), and quit (ESC).

Author: CAIR-UAV Research Team
"""

import os
import sys
import numpy as np

# Suppress Pygame hello message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GRID, RESOURCE


class VisualSimulator:
    """Live Pygame visualization of the UAV mission."""

    def __init__(self, fps: int = 5):
        """Initialize the Pygame window and assets."""
        pygame.init()
        self.fps = fps
        self.cell_size = 30
        self.grid_width = GRID.width
        self.grid_height = GRID.height
        
        # UI panel at the bottom
        self.ui_height = 100
        self.width = self.grid_width * self.cell_size
        self.height = (self.grid_height * self.cell_size) + self.ui_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CAIR-UAV Disaster Response Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)

        self.paused = False
        self.running = True
        self.restart_requested = False

    def handle_events(self) -> bool:
        """Process Pygame events (keyboard/mouse).

        Returns:
            True if simulation should continue, False if quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.restart_requested = True
        return True

    def render(
        self,
        env,
        episode_reward: float,
        step: int,
        action_name: str = "N/A"
    ) -> None:
        """Render the current environment state to the screen."""
        if not self.running:
            return

        self.screen.fill((30, 30, 30))  # Dark background

        # Get state arrays directly from the environment
        uav_pos = env.uav_position
        battery = env._battery
        belief_map = env._belief_map
        hazard_map = env._observed_hazard_map
        visit_count = env._visit_count
        found_survivors = env._found_survivor_cells
        base_pos = GRID.base_position

        # Draw grid
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Base cell color (dark grey, lighter if visited)
                cell_color = [40, 40, 40]
                if visit_count[r, c] > 0:
                    cell_color = [70, 70, 70]
                
                # Add hazard overlay (Red tint)
                if hazard_map[r, c] > 0:
                    hazard_intensity = hazard_map[r, c]
                    cell_color[0] = min(255, cell_color[0] + int(hazard_intensity * 150))
                
                # Add belief overlay (Green tint)
                if belief_map[r, c] > 0.6:
                    belief_intensity = (belief_map[r, c] - 0.5) * 2
                    cell_color[1] = min(255, cell_color[1] + int(belief_intensity * 150))

                pygame.draw.rect(self.screen, tuple(cell_color), rect)
                pygame.draw.rect(self.screen, (20, 20, 20), rect, 1)  # Grid lines

                # Draw Base
                if (r, c) == base_pos:
                    pygame.draw.rect(
                        self.screen,
                        (0, 150, 255),
                        rect.inflate(-10, -10)
                    )

                # Draw Found Survivors
                if (r, c) in found_survivors:
                    pygame.draw.circle(
                        self.screen,
                        (0, 255, 0),
                        (c * self.cell_size + self.cell_size//2, r * self.cell_size + self.cell_size//2),
                        self.cell_size // 3
                    )

        # Draw UAV
        uav_r, uav_c = uav_pos
        uav_rect = pygame.Rect(
            uav_c * self.cell_size,
            uav_r * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, (255, 255, 255), uav_rect.inflate(-6, -6))
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (uav_c * self.cell_size + self.cell_size//2, uav_r * self.cell_size + self.cell_size//2),
            3
        )

        # Draw UI Panel
        ui_rect = pygame.Rect(0, self.grid_height * self.cell_size, self.width, self.ui_height)
        pygame.draw.rect(self.screen, (20, 20, 20), ui_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (0, ui_rect.top), (self.width, ui_rect.top), 2)

        # Battery Bar
        batt_pct = max(0.0, battery / RESOURCE.max_battery)
        batt_color = (0, 255, 0) if batt_pct > 0.5 else (255, 165, 0) if batt_pct > 0.2 else (255, 0, 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, ui_rect.top + 15, 200, 20))
        pygame.draw.rect(self.screen, batt_color, (10, ui_rect.top + 15, int(200 * batt_pct), 20))
        
        batt_text = self.font.render(f"Battery: {battery:.1f}%", True, (255, 255, 255))
        self.screen.blit(batt_text, (220, ui_rect.top + 15))

        # Stats
        stats1 = f"Step: {step} | Reward: {episode_reward:.1f} | Action: {action_name}"
        stats2 = f"Found: {env._survivors_found}/{env._total_survivors} | IG: {env._total_ig:.1f}"
        
        s1_surf = self.font.render(stats1, True, (200, 200, 200))
        s2_surf = self.font.render(stats2, True, (200, 200, 200))
        
        self.screen.blit(s1_surf, (10, ui_rect.top + 45))
        self.screen.blit(s2_surf, (10, ui_rect.top + 70))

        if self.paused:
            pause_surf = self.title_font.render("PAUSED (Space to resume)", True, (255, 255, 0))
            self.screen.blit(pause_surf, (self.width - 250, ui_rect.top + 15))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
