import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

import soundfile as sf

from PIL import Image
from rlr_audio_propagation import Config, Context, ChannelLayout, ChannelLayoutType

class SoundspaceSimulator:
    def __init__(self, glb_file, dest_path, mic_pos=None, sources_pos=[], audio_fmts=["mic"]):
        self.glb_file = glb_file
        self.dest_path = dest_path
        self.audio_fmts = audio_fmts
        
        self.source_spheres = []
        self.mic_positions = []
        self.source_positions = []
        self.adjusted_source_positions = []
        self.scene = trimesh.Scene()

        self.mesh = self.load_and_repair_mesh(self.glb_file)
        
        self.cfg = self.initialize_config()
        
        self.initialize_context(self.cfg)

        if mic_pos is None:
            self.mic_center = self.find_microphone_position()
        else:
            self.mic_center = mic_pos
        self.mic_absolute_positions = self.place_microphones()

        self.source_positions = self.place_sources(sources_pos)
        self.set_listener_sources()
        self.simulate()

    def initialize_config(self):
        return Config()

    def initialize_context(self, config):
        self.ctx = Context(config)
        self.ctx.add_object()
        self.ctx.add_mesh_vertices(self.mesh.vertices.flatten().tolist())
        self.ctx.add_mesh_indices(self.mesh.faces.flatten().tolist(), 3, "default")
        self.ctx.finalize_object_mesh(0)

    def load_and_repair_mesh(self, glb_file):
        mesh = trimesh.load(glb_file, force='mesh')
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        broken_faces = trimesh.repair.broken_faces(new_mesh)
        trimesh.repair.fix_inversion(new_mesh)
        trimesh.repair.fix_normals(new_mesh)
        trimesh.repair.fix_winding(new_mesh)
        new_mesh.fill_holes()
        new_mesh.visual.face_colors = np.ones((len(new_mesh.faces), 4)) * 255
        new_mesh.visual.face_colors[broken_faces] = [255, 0, 0, 255] # make fixed walls red
        self.scene.add_geometry(new_mesh)
        return new_mesh

    def find_microphone_position(self, min_avg_ray_length=3.0, max_attempts=100):
        for attempt in range(max_attempts):
            mic_center = self.get_random_point_inside_mesh(self.mesh)
            avg_ray_length = self.calculate_weighted_average_ray_length(self.mesh, mic_center)
            if avg_ray_length >= min_avg_ray_length:
                print(f"Found suitable microphone position after {attempt+1} attempts")
                return mic_center
        print(f"Could not find a suitable position after {max_attempts} attempts. Using the last attempted position.")
        return mic_center

    def get_random_point_inside_mesh(self, mesh, min_distance_from_surface=0.2):
        while True:
            point = np.random.uniform(mesh.bounds[0], mesh.bounds[1])
            if mesh.contains([point])[0]:
                _, distance, _ = mesh.nearest.on_surface([point])
                if distance[0] >= min_distance_from_surface:
                    return point

    def calculate_weighted_average_ray_length(self, mesh, point, num_rays=100):
        angles = np.random.uniform(0, 2 * np.pi, num_rays)
        elevations = np.random.uniform(-np.pi/2, np.pi/2, num_rays)
        directions = np.column_stack([np.cos(elevations) * np.cos(angles),
                                      np.cos(elevations) * np.sin(angles),
                                      np.sin(elevations)])
        origins = np.tile(point, (num_rays, 1))
        distances = trimesh.proximity.longest_ray(mesh, origins, directions)
        weights = distances ** 2
        weighted_average = np.sum(distances * weights) / np.sum(weights)
        return weighted_average

    def spherical_to_cartesian(self, r, theta, phi):
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        return x, y, z

    def place_microphones(self, mic_radius=0.06):
        mic_positions = [(55, 45), (125, 315), (125, 135), (55, 225)]
        mic_cartesian = [self.spherical_to_cartesian(mic_radius, theta, phi) for theta, phi in mic_positions]
        mic_absolute_positions = [self.mic_center + np.array(pos) for pos in mic_cartesian]
        for mic_pos in mic_absolute_positions:
            self.add_sphere(self.scene, mic_pos, [255, 0, 0], r=0.02)  # Red color for microphones
        return mic_absolute_positions

    def add_sphere(self, scene, pos, color=[0, 0, 0], r=0.2):
        sphere = trimesh.creation.uv_sphere(radius=r)
        sphere.apply_translation(pos)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)
        return sphere

    def place_sources(self, sources_pos):
        source_positions = []
        for source_pos in sources_pos:
            source_positions.append(source_pos)
            self.source_spheres.append(self.add_sphere(self.scene, source_pos, [0, 0, 255], r=0.05))  # Blue for sources
        return source_positions

    def set_listener_sources(self):
        # add listeners (microphones)
        for i, mic_loc in enumerate(self.mic_absolute_positions):
            self.ctx.add_listener(ChannelLayout(ChannelLayoutType.Mono, 1))
            self.ctx.set_listener_position(i, mic_loc.tolist())
        self.ctx.add_source()
        # add sound sources
        for i, position in enumerate(self.source_positions):
            self.ctx.set_source_position(0, position.tolist())  # Source in the environment

    def adjust_sources_elevation(self):
        for i, position in enumerate(self.source_positions):
            new_position = self.adjust_source_elevation(self.mesh, position)
            self.adjusted_source_positions.append(new_position)
            self.source_spheres[i].apply_translation(new_position - position)
            self.ctx.set_source_position(i, new_position.tolist())

    def adjust_source_elevation(self, mesh, position):
        mesh_height = mesh.bounds[1][2] - mesh.bounds[0][2]
        max_elevation_change = mesh_height / 2
        for _ in range(10):
            elevation_change = np.random.uniform(-max_elevation_change, max_elevation_change)
            new_position = position + np.array([0, 0, elevation_change])
            if self.is_point_inside_mesh(mesh, new_position):
                return new_position
        return position

    def is_point_inside_mesh(self, mesh, point):
        return mesh.contains([point])[0]

    def simulate(self):
        self.ctx.simulate()
        efficiency = self.ctx.get_indirect_ray_efficiency()
        print(f"Overall Indirect Ray Efficiency = {efficiency}")

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        room_name = os.path.splitext(os.path.basename(self.glb_file))[0]
        
        vertices = self.mesh.vertices
        ax1.scatter(vertices[:, 0], vertices[:, 1], c='gray', alpha=0.1, s=1)
        ax1.scatter(self.mic_center[0], self.mic_center[1], c='red', s=100, label='Microphone')
        new_sources = np.array(self.source_positions)
        ax1.scatter(new_sources[:, 0], new_sources[:, 1], c='blue', s=25, alpha=0.5, label='Sound Sources')
        ax1.set_title(f'Top-down view of {room_name}')
        ax1.legend()

        ax2.scatter(vertices[:, 0], vertices[:, 2], c='gray', alpha=0.1, s=1)
        ax2.scatter(self.mic_center[0], self.mic_center[2], c='red', s=100, label='Microphone')
        ax2.scatter(new_sources[:, 0], new_sources[:, 2], c='blue', s=25, alpha=0.5, label='Sound Sources')
        ax2.set_title(f'Side view of {room_name}')
        ax2.legend()

        plot_path = self.dest_path / f"{room_name}_plots.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plots saved as {plot_path}")
        
    def generate_rir_data(self):
        sr = int(self.cfg.sample_rate)

        for fmt in self.audio_fmts:
            IRs = []
            coords = []
            max_length = 0
            for source_index, source_position in enumerate(self.source_positions):
                ir_channels = []
    
                max_ir_length = 0
                for listener_index, mic_pos in enumerate(self.mic_absolute_positions):
                    ir_sample_count = self.ctx.get_ir_sample_count(listener_index, source_index)
                    ir_channel_count = self.ctx.get_ir_channel_count(listener_index, source_index)
                    ir = np.zeros((ir_channel_count, ir_sample_count))
                    for i in range(ir_channel_count):
                        channel = np.array(self.ctx.get_ir_channel(listener_index, source_index, i))
                        ir[i] = channel
                    ir_channels.append(ir[0])  # mono channel for each microphone
                    max_ir_length = max(max_ir_length, ir_sample_count)
                
                # Pad all IR channels to the same length
                padded_ir_channels = []
                for ir in ir_channels:
                    padded_ir = np.pad(ir, (0, max_ir_length - len(ir)), mode='constant')
                    padded_ir_channels.append(padded_ir)
                
                combined_ir = np.array(padded_ir_channels)
                if combined_ir.shape[1] > max_length:
                    max_length = combined_ir.shape[1]
                IRs.append(combined_ir)
                
                print(f"IR {source_index}:")
                print(f"  Channels: {combined_ir.shape[0]}")
                print(f"  Samples: {combined_ir.shape[1]}")
                print(f"  Shape: {combined_ir.shape}")
            
            # Pad IRs to max_length
            padded_IRs = []
            for ir in IRs:
                if ir.shape[1] < max_length:
                    padded = np.pad(ir, ((0, 0), (0, max_length - ir.shape[1])), mode='constant')
                    padded_IRs.append(padded)
                else:
                    padded_IRs.append(ir[:, :max_length])
            
            rirs = np.array(padded_IRs)
            # uncomment to write RIR as wav
            #sf.write(self.dest_path.replace(".npy", ".wav"), rirs[0].T, sr)
            np.save(self.dest_path, rirs)

