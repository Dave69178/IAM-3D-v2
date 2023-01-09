import numpy as np


class Environment:
    def __init__(self, boids):
        self._boids = boids
        self._time = 0

    @property
    def boids(self):
        return self._boids

    def get_boid_indiv(self, boid_index):
        return self._boids[boid_index]

    def get_boid_positions(self):
        return np.array([i.position for i in self._boids])

    def get_boid_velocities(self):
        return np.array([i.velocity for i in self._boids])

    @property
    def time(self):
        return self._time

    @time.getter
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    def run(self, model, speed, time_step, max_angle_change_degrees, eta, fov, repulsion_range, orientation_metric_range, orientation_topological_range, attraction_topological_range, attraction_metric_range, max_time):
        def run_topological_metric_hybrid_model():
            for t in range(max_time):
                # Initialise list to store new velocities for simultaneous update of boid positions
                new_velocity_store = []

                # Calculate new velocity for each boid
                for boid in self.boids:
                    # Determine which boids are in each zone
                    # Get distances to other boids from current boid position
                    distance_to_boids = boid.get_distances_to_boids(self)

                    # Get indices of boids within repulsion zone
                    boids_in_repulsion_zone = (distance_to_boids <= repulsion_range).nonzero()[0]

                    if boids_in_repulsion_zone.size != 1:
                        # Boids within repulsion zone (excluding focal boid), hence don't calculate orientation and repulsion interactions
                        # Calculate new intended velocity
                        repulsion_velocity = np.array([0, 0, 0], dtype="float64")
                        for boid_index in boids_in_repulsion_zone:
                            # Add interaction with all in range except the boid in focus
                            if boid_index != boid.index:
                                other_boid = self.get_boid_indiv(boid_index)
                                # Get vector from focal boid to other boid within range
                                vector_from_focus_to_other = other_boid.position - boid.position
                                # Add normalised vector to intended direction
                                repulsion_velocity += - vector_from_focus_to_other / np.linalg.norm(
                                    vector_from_focus_to_other)
                        # Get unit vector for intended direction of motion
                        if np.linalg.norm(repulsion_velocity) == 0:
                            # print("0 rep")
                            unit_repulsion_velocity = boid.velocity
                        else:
                            unit_repulsion_velocity = repulsion_velocity / np.linalg.norm(repulsion_velocity)
                        # Add noise
                        unit_repulsion_velocity = get_velocity_after_adding_noise(unit_repulsion_velocity, eta)
                        # Limit turning angle
                        limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                           unit_repulsion_velocity,
                                                                                           max_angle_change_radians)
                        # Set new velocity for boid
                        new_velocity_store.append(limited_angle_velocity)
                    else:
                        # Initialise velocity vectors for interaction zones
                        orientation_velocity = np.array([0, 0, 0], dtype="float64")
                        attraction_velocity = np.array([0, 0, 0], dtype="float64")
                        unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")
                        unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")
                        # Get indices of boids within orientation metric and topological range
                        boids_within_metric_range = ((distance_to_boids <= orientation_metric_range) & (
                                    distance_to_boids > repulsion_range)).nonzero()[0]
                        if boids_within_metric_range.size != 0:
                            # Boids within orientation metric range (Excluding focal boid), hence calculate effect of interactions
                            if boids_within_metric_range.size <= orientation_topological_range:
                                # Topological range not exceeded within metric range, so calculate intended velocity with all boids in metric range
                                for boid_index in boids_within_metric_range:
                                    # Add interaction with all in range
                                    other_boid = self.get_boid_indiv(boid_index)
                                    # Check bird is within fov
                                    if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                        # Get velocity of other boid
                                        velocity_of_other = other_boid.velocity
                                        # Add normalised vector to intended direction
                                        orientation_velocity += velocity_of_other / np.linalg.norm(velocity_of_other)
                                if np.linalg.norm(orientation_velocity) != 0:
                                    # Get unit vector for direction relating to orientation interactions
                                    unit_orientation_velocity = orientation_velocity / np.linalg.norm(
                                        orientation_velocity)
                                else:
                                    unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")
                            else:
                                # Find the nearest orientation_topological_range number of birds within metric range
                                distances_within_metric_range = distance_to_boids[
                                    (distance_to_boids <= orientation_metric_range) & (
                                                distance_to_boids > repulsion_range)]
                                # Gets indices of boids within topological and metric range
                                indices_that_sort_distances = np.argsort(distances_within_metric_range)[
                                                              0:orientation_topological_range]
                                boids_within_both_ranges = boids_within_metric_range[indices_that_sort_distances]
                                # Calculate intended velocity
                                orientation_velocity = np.array([0, 0, 0], dtype="float64")
                                for boid_index in boids_within_both_ranges:
                                    # Add interaction with all in range
                                    other_boid = self.get_boid_indiv(boid_index)
                                    # Check bird is within fov
                                    if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                        # Get velocity of other boid
                                        velocity_of_other = other_boid.velocity
                                        # Add normalised vector to intended direction
                                        orientation_velocity += velocity_of_other / np.linalg.norm(velocity_of_other)
                                # Get unit vector for direction relating to orientation interactions
                                if np.linalg.norm(orientation_velocity) != 0:
                                    unit_orientation_velocity = orientation_velocity / np.linalg.norm(
                                        orientation_velocity)
                                else:
                                    # print("0 ori")
                                    unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")

                        # Get indices of boids within attraction range
                        boids_outside_orientation_metric_range = \
                        (distance_to_boids > orientation_metric_range).nonzero()[0]
                        if boids_outside_orientation_metric_range.size != 0:
                            if boids_outside_orientation_metric_range.size <= attraction_topological_range:
                                # Topological range not exceeded, so calculate intended velocity with all boids
                                for boid_index in boids_outside_orientation_metric_range:
                                    # Add interaction with all in zone (Don't need to check for focal bird as they will never be included)
                                    other_boid = self.get_boid_indiv(boid_index)
                                    # Check bird is within fov
                                    if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                        # Get vector from focal boid to other boid within range
                                        vector_from_focus_to_other = other_boid.position - boid.position
                                        # Add normalised vector to intended direction
                                        attraction_velocity += vector_from_focus_to_other / np.linalg.norm(
                                            vector_from_focus_to_other)
                                if np.linalg.norm(attraction_velocity) != 0:
                                    # Get unit vector for direction relating to attraction interactions
                                    unit_attraction_velocity = attraction_velocity / np.linalg.norm(attraction_velocity)
                                else:
                                    # print("0 attr")
                                    unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")
                            else:
                                # Find the nearest attraction_topological_range number of birds
                                distances_to_boids_outside_orientation_metric_range = distance_to_boids[
                                    distance_to_boids > orientation_metric_range]
                                # Gets indices of boids within topological range
                                indices_that_sort_distances = np.argsort(
                                    distances_to_boids_outside_orientation_metric_range)[0:attraction_topological_range]
                                boids_within_attraction_zone = boids_outside_orientation_metric_range[
                                    indices_that_sort_distances]
                                # Calculate intended velocity from interactions with attraction zone
                                attraction_velocity = np.array([0, 0, 0], dtype="float64")
                                for boid_index in boids_within_attraction_zone:
                                    # Add interaction with all in zone (Don't need to check for focal bird as they will never be included)
                                    other_boid = self.get_boid_indiv(boid_index)
                                    # Check bird is within fov
                                    if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                        # Get vector from focal boid to other boid within range
                                        vector_from_focus_to_other = other_boid.position - boid.position
                                        # Add normalised vector to intended direction
                                        attraction_velocity += vector_from_focus_to_other / np.linalg.norm(
                                            vector_from_focus_to_other)
                                # Get unit vector for direction relating to attraction interactions
                                if np.linalg.norm(attraction_velocity) != 0:
                                    unit_attraction_velocity = attraction_velocity / np.linalg.norm(attraction_velocity)
                                else:
                                    unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")

                        # Calculate new intended velocity
                        if np.any(unit_orientation_velocity):
                            if np.any(unit_attraction_velocity):
                                # Orientation and attraction both contribute
                                new_velocity = 1 / 2 * (unit_orientation_velocity + unit_attraction_velocity)
                                new_unit_velocity = new_velocity / np.linalg.norm(new_velocity)
                                # Add noise
                                new_unit_velocity = get_velocity_after_adding_noise(new_unit_velocity, eta)
                                # Limit turning angle
                                limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                                   new_unit_velocity,
                                                                                                   max_angle_change_radians)
                                new_velocity_store.append(limited_angle_velocity)
                            else:
                                # Orientation interaction only
                                # Add noise
                                unit_orientation_velocity = get_velocity_after_adding_noise(unit_orientation_velocity,
                                                                                            eta)
                                # Limit turning angle
                                limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                                   unit_orientation_velocity,
                                                                                                   max_angle_change_radians)
                                new_velocity_store.append(limited_angle_velocity)
                        elif np.any(unit_attraction_velocity):
                            # Attraction only
                            # Add noise
                            unit_attraction_velocity = get_velocity_after_adding_noise(unit_attraction_velocity, eta)
                            # Limit turning angle
                            limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                               unit_attraction_velocity,
                                                                                               max_angle_change_radians)
                            new_velocity_store.append(limited_angle_velocity)
                        else:
                            # No interactions, keep same velocity as previous timestep
                            # Add noise
                            new_unit_velocity = get_velocity_after_adding_noise(boid.velocity, eta)
                            new_velocity_store.append(new_unit_velocity)

                # Update boid velocities
                for boid in self.boids:
                    boid.velocity = new_velocity_store[boid.index]

                # Update boid positions
                for boid in self.boids:
                    boid.position += speed * time_step * boid.velocity

                # Increment time
                self.time += 1

                yield None

        def run_metric_model():
            for t in range(max_time):
                # Initialise list to store new velocities for simultaneous update of boid positions
                new_velocity_store = []

                # Calculate new velocity for each boid
                for boid in self.boids:
                    # Determine which boids are in each zone
                    # Get distances to other boids from current boid position
                    distance_to_boids = boid.get_distances_to_boids(self)

                    # Get indices of boids within repulsion zone
                    boids_in_repulsion_zone = (distance_to_boids <= repulsion_range).nonzero()[0]

                    if boids_in_repulsion_zone.size != 1:
                        # Boids within repulsion zone (excluding focal boid), hence don't calculate orientation and repulsion interactions
                        # Calculate new intended velocity
                        repulsion_velocity = np.array([0, 0, 0], dtype="float64")
                        for boid_index in boids_in_repulsion_zone:
                            # Add interaction with all in range except the boid in focus
                            if boid_index != boid.index:
                                other_boid = self.get_boid_indiv(boid_index)
                                # Get vector from focal boid to other boid within range
                                vector_from_focus_to_other = other_boid.position - boid.position
                                # Add normalised vector to intended direction
                                repulsion_velocity += - vector_from_focus_to_other / np.linalg.norm(
                                    vector_from_focus_to_other)
                        # Get unit vector for intended direction of motion
                        if np.linalg.norm(repulsion_velocity) == 0:
                            # print("0 rep")
                            unit_repulsion_velocity = boid.velocity
                        else:
                            unit_repulsion_velocity = repulsion_velocity / np.linalg.norm(repulsion_velocity)
                        # Add noise
                        unit_repulsion_velocity = get_velocity_after_adding_noise(unit_repulsion_velocity, eta)
                        # Limit turning angle
                        limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                           unit_repulsion_velocity,
                                                                                           max_angle_change_radians)
                        # Set new velocity for boid
                        new_velocity_store.append(limited_angle_velocity)
                    else:
                        # Initialise velocity vectors for interaction zones
                        orientation_velocity = np.array([0, 0, 0], dtype="float64")
                        attraction_velocity = np.array([0, 0, 0], dtype="float64")
                        unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")
                        unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")
                        # Get indices of boids within orientation metric range
                        boids_within_metric_range = ((distance_to_boids <= orientation_metric_range) & (
                                    distance_to_boids > repulsion_range)).nonzero()[0]
                        if boids_within_metric_range.size != 0:
                            # Boids within orientation metric range (Excluding focal boid), hence calculate effect of interactions
                            for boid_index in boids_within_metric_range:
                                # Add interaction with all in range
                                other_boid = self.get_boid_indiv(boid_index)
                                # Check bird is within fov
                                if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                    # Get velocity of other boid
                                    velocity_of_other = other_boid.velocity
                                    # Add normalised vector to intended direction
                                    orientation_velocity += velocity_of_other / np.linalg.norm(velocity_of_other)
                            if np.linalg.norm(orientation_velocity) != 0:
                                # Get unit vector for direction relating to orientation interactions
                                unit_orientation_velocity = orientation_velocity / np.linalg.norm(
                                    orientation_velocity)
                            else:
                                unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")

                        # Get indices of boids within attraction range
                        boids_outside_orientation_metric_range = \
                        ((distance_to_boids > orientation_metric_range) & (distance_to_boids <= attraction_metric_range)).nonzero()[0]
                        if boids_outside_orientation_metric_range.size != 0:
                            for boid_index in boids_outside_orientation_metric_range:
                                # Add interaction with all in zone (Don't need to check for focal bird as they will never be included)
                                other_boid = self.get_boid_indiv(boid_index)
                                # Check bird is within fov
                                if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                    # Get vector from focal boid to other boid within range
                                    vector_from_focus_to_other = other_boid.position - boid.position
                                    # Add normalised vector to intended direction
                                    attraction_velocity += vector_from_focus_to_other / np.linalg.norm(
                                        vector_from_focus_to_other)
                            if np.linalg.norm(attraction_velocity) != 0:
                                # Get unit vector for direction relating to attraction interactions
                                unit_attraction_velocity = attraction_velocity / np.linalg.norm(attraction_velocity)
                            else:
                                # print("0 attr")
                                unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")

                        # Calculate new intended velocity
                        if np.any(unit_orientation_velocity):
                            if np.any(unit_attraction_velocity):
                                # Orientation and attraction both contribute
                                new_velocity = 1 / 2 * (unit_orientation_velocity + unit_attraction_velocity)
                                new_unit_velocity = new_velocity / np.linalg.norm(new_velocity)
                                # Add noise
                                new_unit_velocity = get_velocity_after_adding_noise(new_unit_velocity, eta)
                                # Limit turning angle
                                limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                                   new_unit_velocity,
                                                                                                   max_angle_change_radians)
                                new_velocity_store.append(limited_angle_velocity)
                            else:
                                # Orientation interaction only
                                # Add noise
                                unit_orientation_velocity = get_velocity_after_adding_noise(unit_orientation_velocity,
                                                                                            eta)
                                # Limit turning angle
                                limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                                   unit_orientation_velocity,
                                                                                                   max_angle_change_radians)
                                new_velocity_store.append(limited_angle_velocity)
                        elif np.any(unit_attraction_velocity):
                            # Attraction only
                            # Add noise
                            unit_attraction_velocity = get_velocity_after_adding_noise(unit_attraction_velocity, eta)
                            # Limit turning angle
                            limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                               unit_attraction_velocity,
                                                                                               max_angle_change_radians)
                            new_velocity_store.append(limited_angle_velocity)
                        else:
                            # No interactions, keep same velocity as previous timestep
                            # Add noise
                            new_unit_velocity = get_velocity_after_adding_noise(boid.velocity, eta)
                            new_velocity_store.append(new_unit_velocity)

                # Update boid velocities
                for boid in self.boids:
                    boid.velocity = new_velocity_store[boid.index]

                # Update boid positions
                for boid in self.boids:
                    boid.position += speed * time_step * boid.velocity

                # Increment time
                self.time += 1

                yield None

        def run_topological_model():
            for t in range(max_time):
                # Initialise list to store new velocities for simultaneous update of boid positions
                new_velocity_store = []

                # Calculate new velocity for each boid
                for boid in self.boids:
                    # Determine which boids are in each zone
                    # Get distances to other boids from current boid position
                    distance_to_boids = boid.get_distances_to_boids(self)

                    # Get indices of boids within repulsion zone
                    boids_in_repulsion_zone = (distance_to_boids <= repulsion_range).nonzero()[0]

                    if boids_in_repulsion_zone.size != 1:
                        # Boids within repulsion zone (excluding focal boid), hence don't calculate orientation and repulsion interactions
                        # Calculate new intended velocity
                        repulsion_velocity = np.array([0, 0, 0], dtype="float64")
                        for boid_index in boids_in_repulsion_zone:
                            # Add interaction with all in range except the boid in focus
                            if boid_index != boid.index:
                                other_boid = self.get_boid_indiv(boid_index)
                                # Get vector from focal boid to other boid within range
                                vector_from_focus_to_other = other_boid.position - boid.position
                                # Add normalised vector to intended direction
                                repulsion_velocity += - vector_from_focus_to_other / np.linalg.norm(
                                    vector_from_focus_to_other)
                        # Get unit vector for intended direction of motion
                        if np.linalg.norm(repulsion_velocity) == 0:
                            # print("0 rep")
                            unit_repulsion_velocity = boid.velocity
                        else:
                            unit_repulsion_velocity = repulsion_velocity / np.linalg.norm(repulsion_velocity)
                        # Add noise
                        unit_repulsion_velocity = get_velocity_after_adding_noise(unit_repulsion_velocity, eta)
                        # Limit turning angle
                        limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                           unit_repulsion_velocity,
                                                                                           max_angle_change_radians)
                        # Set new velocity for boid
                        new_velocity_store.append(limited_angle_velocity)
                    else:
                        # Initialise velocity vectors for interaction zones
                        unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")
                        unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")
                        # Get indices to all other boids
                        other_boids = distance_to_boids.nonzero()[0]
                        if other_boids.size != 0:
                            # Find the nearest orientation_topological_range number of birds
                            # Gets indices of boids within topological range
                            indices_that_sort_distances = np.argsort(distance_to_boids[distance_to_boids > 0])[
                                                          0:orientation_topological_range]
                            boids_within_range = other_boids[indices_that_sort_distances]
                            # Calculate intended velocity
                            orientation_velocity = np.array([0, 0, 0], dtype="float64")
                            for boid_index in boids_within_range:
                                # Add interaction with all in range
                                other_boid = self.get_boid_indiv(boid_index)
                                # Check bird is within fov
                                if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                    # Get velocity of other boid
                                    velocity_of_other = other_boid.velocity
                                    # Add normalised vector to intended direction
                                    orientation_velocity += velocity_of_other / np.linalg.norm(velocity_of_other)
                                # Setting distance to this boid to zero so that it is removed from attraction range selection
                                distance_to_boids[boid_index] = 0
                            # Get unit vector for direction relating to orientation interactions
                            if np.linalg.norm(orientation_velocity) != 0:
                                unit_orientation_velocity = orientation_velocity / np.linalg.norm(
                                    orientation_velocity)
                            else:
                                # print("0 ori")
                                unit_orientation_velocity = np.array([0, 0, 0], dtype="float64")

                        # Get indices of boids within attraction range
                        boids_outside_orientation_topological_range = (distance_to_boids > 0).nonzero()[0]
                        if boids_outside_orientation_topological_range.size != 0:
                            # Find the nearest attraction_topological_range number of birds
                            distances_to_boids_outside_orientation_topological_range = distance_to_boids[boids_outside_orientation_topological_range]
                            # Gets indices of boids within topological range
                            indices_that_sort_distances = np.argsort(
                                distances_to_boids_outside_orientation_topological_range)[0:attraction_topological_range]
                            boids_within_attraction_zone = boids_outside_orientation_topological_range[
                                indices_that_sort_distances]
                            # Calculate intended velocity from interactions with attraction zone
                            attraction_velocity = np.array([0, 0, 0], dtype="float64")
                            for boid_index in boids_within_attraction_zone:
                                # Add interaction with all in zone (Don't need to check for focal bird as they will never be included)
                                other_boid = self.get_boid_indiv(boid_index)
                                # Check bird is within fov
                                if is_boid_in_fov(boid, other_boid, cos_half_fov):
                                    # Get vector from focal boid to other boid within range
                                    vector_from_focus_to_other = other_boid.position - boid.position
                                    # Add normalised vector to intended direction
                                    attraction_velocity += vector_from_focus_to_other / np.linalg.norm(
                                        vector_from_focus_to_other)
                            # Get unit vector for direction relating to attraction interactions
                            if np.linalg.norm(attraction_velocity) != 0:
                                unit_attraction_velocity = attraction_velocity / np.linalg.norm(attraction_velocity)
                            else:
                                unit_attraction_velocity = np.array([0, 0, 0], dtype="float64")

                        # Calculate new intended velocity
                        if np.any(unit_orientation_velocity):
                            if np.any(unit_attraction_velocity):
                                # Orientation and attraction both contribute
                                new_velocity = 1 / 2 * (unit_orientation_velocity + unit_attraction_velocity)
                                new_unit_velocity = new_velocity / np.linalg.norm(new_velocity)
                                # Add noise
                                new_unit_velocity = get_velocity_after_adding_noise(new_unit_velocity, eta)
                                # Limit turning angle
                                limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                                   new_unit_velocity,
                                                                                                   max_angle_change_radians)
                                new_velocity_store.append(limited_angle_velocity)
                            else:
                                # Orientation interaction only
                                # Add noise
                                unit_orientation_velocity = get_velocity_after_adding_noise(unit_orientation_velocity,
                                                                                            eta)
                                # Limit turning angle
                                limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                                   unit_orientation_velocity,
                                                                                                   max_angle_change_radians)
                                new_velocity_store.append(limited_angle_velocity)
                        elif np.any(unit_attraction_velocity):
                            # Attraction only
                            # Add noise
                            unit_attraction_velocity = get_velocity_after_adding_noise(unit_attraction_velocity, eta)
                            # Limit turning angle
                            limited_angle_velocity = get_velocity_after_limiting_turning_angle(boid.velocity,
                                                                                               unit_attraction_velocity,
                                                                                               max_angle_change_radians)
                            new_velocity_store.append(limited_angle_velocity)
                        else:
                            # No interactions, keep same velocity as previous timestep
                            # Add noise
                            new_unit_velocity = get_velocity_after_adding_noise(boid.velocity, eta)
                            new_velocity_store.append(new_unit_velocity)

                # Update boid velocities
                for boid in self.boids:
                    boid.velocity = new_velocity_store[boid.index]

                # Update boid positions
                for boid in self.boids:
                    boid.position += speed * time_step * boid.velocity

                # Increment time
                self.time += 1

                yield None

        max_angle_change_radians = max_angle_change_degrees * (np.pi / 180)
        fov_radians = fov * (np.pi / 180)
        cos_half_fov = np.cos(fov_radians / 2)
        if model == "hybrid":
            yield from run_topological_metric_hybrid_model()
        elif model == "metric":
            yield from run_metric_model()
        elif model == "topological":
            yield from run_topological_model()


def get_velocity_after_limiting_turning_angle(current_velocity, new_direction, max_angle_change_radians):
    dot_product = np.dot(current_velocity, new_direction)
    # Check for floating point error
    if dot_product > 1:
        angle_change = 0
    else:
        angle_change = np.arccos(dot_product)

    if angle_change <= max_angle_change_radians:
        return new_direction
    else:
        a = new_direction - current_velocity * np.dot(current_velocity, new_direction)
        a = a / np.linalg.norm(a)
        return current_velocity * np.cos(max_angle_change_radians) + a * np.sin(max_angle_change_radians)


def get_velocity_after_adding_noise(intended_direction, noise_stddev):
    # Generate random perturbation
    random_alpha = np.random.uniform(0, 2 * np.pi)
    random_psi = np.abs(np.random.normal(0, noise_stddev))
    random_coords = np.array([np.cos(random_alpha) * np.sin(random_psi), np.sin(random_alpha) * np.sin(random_psi), np.cos(random_psi)])

    # Convert to spherical coords and apply perturbation
    colatitude = np.arccos(intended_direction[2])
    if colatitude == 0 or colatitude == np.pi:
        azimuthal_angle = 0
    else:
        azimuthal_angle = np.arctan2(intended_direction[1], intended_direction[0])

    transformation_mat = np.array([[np.cos(colatitude) * np.cos(azimuthal_angle), -np.sin(azimuthal_angle),
                                    np.sin(colatitude) * np.cos(azimuthal_angle)],
                                   [np.cos(colatitude) * np.sin(azimuthal_angle), np.cos(azimuthal_angle),
                                    np.sin(colatitude) * np.sin(azimuthal_angle)],
                                   [-np.sin(colatitude), 0, np.cos(colatitude)]])
    return np.matmul(transformation_mat, random_coords)


def is_boid_in_fov(focal_boid, other_boid, cos_half_fov):
    # Check if boid is in field of view
    # h is heading of focal bird, p is position of focal bird, q is position of other bird
    h_minus_p = focal_boid.velocity
    q_minus_p = other_boid.position - focal_boid.position
    cos_of_angle_between_heading_and_other_boid_position = np.dot(h_minus_p, q_minus_p) / (np.linalg.norm(h_minus_p) * np.linalg.norm(q_minus_p))
    if cos_of_angle_between_heading_and_other_boid_position < cos_half_fov:
        # Other boid is outside fov
        return False
    else:
        return True
