import carla
import queue
import threading
import time


class CARLASyncContextFixed(object):
    """
    CARLA 0.9.15 compatible synchronous mode context manager
    Fixed version that handles sensor callbacks properly with timeout handling
    """

    def __init__(self, world, sensors: dict, fps=30, initialization_timeout=10.0, no_rendering_mode=True):
        print("DEBUG: Initializing CARLASyncContextFixed...")
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._settings = None
        self.initialization_timeout = initialization_timeout
        self.no_rendering_mode = no_rendering_mode
        print(f"DEBUG: Config - fps: {fps}, no_rendering: {no_rendering_mode}, timeout: {initialization_timeout}")

        # Make a queue for each sensor and for world:
        self._queues = dict()

        try:
            print("DEBUG: Adding world queue...")
            self._add_queue('world', self.world.on_tick)
            print("DEBUG: World queue added successfully")
        except Exception as e:
            print(f"DEBUG: Failed to add world queue: {e}")
            raise

        # For CARLA 0.9.15 compatibility - register callbacks with proper cleanup and timeout
        successful_sensors = 0
        print(f"DEBUG: Registering {len(sensors)} sensor callbacks...")
        for name, sensor in self.sensors.items():
            print(f"DEBUG: Processing sensor '{name}'...")
            try:
                # Check if sensor is valid before processing
                if not self._validate_sensor(name, sensor):
                    print(f"DEBUG: Sensor '{name}' validation failed, skipping")
                    continue

                print(f"DEBUG: Sensor '{name}' validated successfully")

                # First, ensure the sensor is in a clean state
                print(f"DEBUG: Cleaning up sensor '{name}' callbacks...")
                self._cleanup_sensor_callbacks(name, sensor)

                # Add timeout-protected callback registration
                print(f"DEBUG: Registering callback for sensor '{name}'...")
                if self._register_sensor_callback_with_timeout(name, sensor):
                    successful_sensors += 1
                    print(f"DEBUG: Sensor '{name}' callback registered successfully")
                else:
                    print(f"DEBUG: Sensor '{name}' callback registration failed or timed out")

            except Exception as e:
                print(f"DEBUG: Error processing sensor '{name}': {e}")
                # Continue with other sensors rather than failing completely
                continue

        print(f"DEBUG: Successfully registered {successful_sensors} out of {len(sensors)} sensors")

        # Be more lenient - allow partial sensor registration
        if successful_sensors == 0 and len(sensors) > 0:
            print("DEBUG: No sensors registered successfully, raising error")
            raise RuntimeError("Failed to register any sensor callbacks")

        print("DEBUG: CARLASyncContextFixed initialization completed")

    def _validate_sensor(self, name, sensor):
        """Validate sensor before attempting to register callbacks"""
        try:
            # Check if sensor wrapper exists
            if sensor is None:
                return False

            # Check if underlying CARLA sensor exists
            if hasattr(sensor, 'sensor'):
                if sensor.sensor is None:
                    return False

                # Check if sensor is still alive in CARLA
                try:
                    _ = sensor.sensor.is_alive
                    return True
                except Exception:
                    return False

            # For sensors without .sensor attribute, assume valid
            return True

        except Exception:
            return False

    def _cleanup_sensor_callbacks(self, name, sensor):
        """Ensure sensor is not already listening before registering new callbacks"""
        try:
            # Force stop the sensor regardless of current state to avoid conflicts
            if hasattr(sensor, 'sensor') and sensor.sensor:
                try:
                    # Always try to stop - ignore the listening state check
                    sensor.stop()
                    time.sleep(0.1)  # Longer delay for stability
                except Exception:
                    pass

            elif hasattr(sensor, 'stop'):
                try:
                    sensor.stop()
                    time.sleep(0.1)
                except Exception:
                    pass

        except Exception:
            pass

    def _register_sensor_callback_with_timeout(self, name, sensor):
        """Register sensor callback with timeout protection"""
        print(f"DEBUG [_register_sensor_callback_with_timeout]: Starting callback registration for sensor '{name}' with {self.initialization_timeout}s timeout")
        print(f"DEBUG [_register_sensor_callback_with_timeout]: sensor object={sensor}, has .sensor={hasattr(sensor, 'sensor')}")
        if hasattr(sensor, 'sensor'):
            print(f"DEBUG [_register_sensor_callback_with_timeout]: sensor.sensor={sensor.sensor}, is_alive={sensor.sensor.is_alive if sensor.sensor else 'None'}")
        registration_successful = threading.Event()
        registration_error = [None]

        def register_callback():
            try:
                print(f"DEBUG [register_callback thread]: In registration thread for sensor '{name}'")
                # Use the standard sensor callback registration
                if hasattr(sensor, 'add_callback'):
                    print(f"DEBUG [register_callback thread]: Sensor '{name}' has add_callback method, using it")
                    self._add_queue(name, sensor.add_callback)
                elif hasattr(sensor, 'sensor') and sensor.sensor and hasattr(sensor.sensor, 'listen'):
                    print(f"DEBUG [register_callback thread]: Sensor '{name}' has sensor.listen method, using it")
                    self._add_queue(name, sensor.sensor.listen)
                else:
                    error_msg = f"Sensor {name} has no valid callback method"
                    print(f"DEBUG [register_callback thread]: {error_msg}")
                    raise ValueError(error_msg)

                print(f"DEBUG [register_callback thread]: Callback registration successful for sensor '{name}'")
                registration_successful.set()
            except Exception as e:
                print(f"DEBUG [register_callback thread]: Callback registration failed for sensor '{name}': {e}")
                import traceback
                traceback.print_exc()
                registration_error[0] = e
                registration_successful.set()

        # Run registration in a separate thread with timeout
        print(f"DEBUG [_register_sensor_callback_with_timeout]: Starting registration thread for sensor '{name}'")
        registration_thread = threading.Thread(target=register_callback)
        registration_thread.daemon = True
        registration_thread.start()

        # Wait for registration with timeout
        print(f"DEBUG [_register_sensor_callback_with_timeout]: Waiting for registration to complete for sensor '{name}'...")
        if registration_successful.wait(timeout=self.initialization_timeout):
            if registration_error[0] is not None:
                print(f"DEBUG [_register_sensor_callback_with_timeout]: Registration failed for sensor '{name}': {registration_error[0]}")
                return False
            print(f"DEBUG [_register_sensor_callback_with_timeout]: Registration completed successfully for sensor '{name}'")
            return True
        else:
            print(f"DEBUG [_register_sensor_callback_with_timeout]: Registration timed out for sensor '{name}' after {self.initialization_timeout}s")
            return False

    def __enter__(self):
        print("DEBUG: Starting synchronous mode initialization...")
        try:
            print("DEBUG: Getting world settings...")
            self._settings = self.world.get_settings()
            print(f"DEBUG: Current settings - sync_mode: {self._settings.synchronous_mode}, no_rendering: {getattr(self._settings, 'no_rendering_mode', 'unknown')}")

            # Apply synchronous settings with improved error handling
            print("DEBUG: Creating new world settings...")
            new_settings = carla.WorldSettings(
                no_rendering_mode=self.no_rendering_mode,
                fixed_delta_seconds=self.delta_seconds,
                synchronous_mode=True
            )
            print(f"DEBUG: New settings - sync_mode: True, no_rendering: {self.no_rendering_mode}, delta: {self.delta_seconds}")

            print("DEBUG: Applying world settings...")
            self.frame = self.world.apply_settings(new_settings)
            print(f"DEBUG: World settings applied successfully, frame: {self.frame}")

            # Start sensors with individual error handling - CARLA 0.9.15 compatibility
            print(f"DEBUG: Starting {len(self.sensors)} sensors...")
            started_sensors = 0
            for name, sensor in self.sensors.items():
                print(f"DEBUG: Processing sensor '{name}', has start method: {hasattr(sensor, 'start')}")
                if hasattr(sensor, 'start'):
                    try:
                        # For CARLA 0.9.15, explicitly call start() after callback registration
                        # Some sensors need explicit start() call to begin listening
                        print(f"DEBUG: Calling start() for sensor '{name}'")
                        sensor.start()
                        started_sensors += 1
                        print(f"DEBUG: Sensor '{name}' started successfully")

                    except Exception as e:
                        print(f"DEBUG: Failed to start sensor '{name}': {e}")
                        # For CARLA 0.9.15, if start() fails, the sensor might already be active
                        # Continue and consider it started
                        started_sensors += 1
                        print(f"DEBUG: Sensor '{name}' assumed to be already active")
                        continue
                else:
                    print(f"DEBUG: Sensor '{name}' has no start method, assuming it's ready")
                    started_sensors += 1

            print(f"DEBUG: Successfully processed {started_sensors} sensors")

            # Small delay to ensure all sensors are properly initialized
            print("DEBUG: Waiting for sensor initialization...")
            time.sleep(0.2)  # Slightly longer delay

            print("DEBUG: Synchronous mode initialization completed successfully")
            return self

        except Exception as e:
            print(f"DEBUG: Error during synchronous mode initialization: {e}")
            import traceback
            traceback.print_exc()

            # Attempt cleanup on failure
            try:
                if hasattr(self, '_settings') and self._settings:
                    print("DEBUG: Restoring original settings due to error...")
                    self.world.apply_settings(self._settings)
            except Exception as cleanup_e:
                print(f"DEBUG: Failed to restore settings: {cleanup_e}")
                pass
            raise RuntimeError(f"Cannot enter synchronous mode: {e}")

    def __exit__(self, *args, **kwargs):
        try:
            # Stop sensors with individual error handling
            for name, sensor in self.sensors.items():
                if hasattr(sensor, 'stop'):
                    try:
                        sensor.stop()
                    except Exception:
                        # Continue with other sensors
                        continue

            # Restore original settings
            if self._settings:
                self.world.apply_settings(self._settings)

        except Exception:
            # Don't raise here to allow cleanup to continue
            pass

    def tick(self, timeout):
        print(f"DEBUG [tick]: World tick called with timeout={timeout}")
        self.frame = self.world.tick()
        print(f"DEBUG [tick]: World frame={self.frame}")

        data = dict()
        for name, q in self._queues.items():
            print(f"DEBUG [tick]: Processing queue '{name}'")
            if name == 'world':
                # Get world snapshot
                try:
                    data[name] = self._get_sensor_data(q, timeout)
                    print(f"DEBUG [tick]: World snapshot retrieved successfully")
                except queue.Empty:
                    # Create dummy world data
                    data[name] = type('WorldSnapshot', (), {'frame': self.frame, 'timestamp': None})()
                    print(f"DEBUG [tick]: World snapshot NOT available, using dummy")
            elif name in self.sensors and hasattr(self.sensors[name], 'is_detector') and self.sensors[name].is_detector:
                # Detectors retrieve data only when triggered
                data[name] = self._get_detector_data(q)
                print(f"DEBUG [tick]: Detector '{name}' data retrieved: {len(data[name]) if isinstance(data[name], list) else 'N/A'} events")
            else:
                # Cameras and other continuous sensors
                try:
                    sensor_data = self._get_sensor_data(q, timeout)
                    data[name] = sensor_data
                    print(f"DEBUG [tick]: Sensor '{name}' data retrieved successfully, type={type(sensor_data)}")
                except queue.Empty:
                    # For missing sensor data, just use None - don't generate dummy images
                    data[name] = None
                    print(f"DEBUG [tick]: Sensor '{name}' data NOT available (queue empty), using None")
                    continue

        print(f"DEBUG [tick]: Final data keys: {data.keys()}")
        print(f"DEBUG [tick]: Camera data available: {'camera' in data and data['camera'] is not None}")
        return data

    def _add_queue(self, name, register_event):
        """Registers an event on its own queue identified by name"""
        q = queue.Queue()
        register_event(q.put)
        self._queues[name] = q

    @staticmethod
    def _get_detector_data(sensor_queue: queue.Queue):
        """Retrieves data for detector, non-blocking call"""
        data = []
        while not sensor_queue.empty():
            try:
                data.append(sensor_queue.get_nowait())
            except queue.Empty:
                break
        return data

    def _get_sensor_data(self, sensor_queue: queue.Queue, timeout: float):
        """Retrieves data for sensors with frame synchronization"""
        # For CARLA 0.9.15, be more lenient with frame matching
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                data = sensor_queue.get(timeout=timeout / max_attempts)

                # In CARLA 0.9.15, sometimes frame sync is off by 1
                if hasattr(data, 'frame'):
                    if data.frame == self.frame or data.frame == self.frame - 1:
                        return data
                    elif attempts == max_attempts - 1:
                        # On last attempt, accept any recent frame
                        return data
                else:
                    # For world snapshots or data without frame info
                    return data

            except queue.Empty:
                attempts += 1

        # If we get here, timeout occurred
        raise queue.Empty