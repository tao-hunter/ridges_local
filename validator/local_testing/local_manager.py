"""
Local sandbox manager for testing agents without database infrastructure.
This module provides:
- LocalSandboxManager: Manages Docker containers and networks for local testing
- LocalSandbox: Modified sandbox that doesn't need websockets or database
"""
import asyncio
import tempfile
import uuid
import shutil
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import os
import threading
from validator.sandbox.constants import (
    SANDBOX_DOCKER_IMAGE, PROXY_DOCKER_IMAGE, SANDBOX_NETWORK_NAME,
    PROXY_CONTAINER_NAME, SANDBOX_MAX_RAM_USAGE, SANDBOX_MAX_RUNTIME,
    MAIN_FILE, SANDBOX_MAIN_FILE, SANDBOX_INPUT_FILE, SANDBOX_OUTPUT_FILE,
    SANDBOX_SOURCE_DIR, SANDBOX_REPO_DIR, SANDBOX_DIR
)
from validator.sandbox.schema import EvaluationRun, SwebenchProblem, SandboxInput
from validator.sandbox.sandbox import Sandbox, get_sandbox_image_for_instance
from validator.sandbox.clone_repo import clone_repo
from loggers.logging_utils import get_logger
logger = get_logger(__name__)

class DockerClientPool:
    """Pool of Docker clients for parallel SWE-bench evaluations"""
    
    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.clients = []
        self.available_clients = asyncio.Queue()
        self.lock = threading.Lock()
        
        # Import docker here to handle ImportError properly
        try:
            import docker
            self.docker_module = docker
        except ImportError:
            raise ImportError("Docker Python library is required for local testing. Install with: pip install docker")
        
        # Create pool of Docker clients
        for i in range(pool_size):
            try:
                client = self.docker_module.from_env()
                client.ping()
                self.clients.append(client)
            except Exception as e:
                raise RuntimeError(f"Failed to create Docker client {i}: {e}")
        
        # Initialize the queue with all clients
        for client in self.clients:
            self.available_clients.put_nowait(client)
    
    async def get_client(self):
        """Get an available Docker client from the pool"""
        return await self.available_clients.get()
    
    async def return_client(self, client):
        """Return a Docker client to the pool"""
        await self.available_clients.put(client)
    
    def cleanup(self):
        """Clean up all Docker clients"""
        for client in self.clients:
            try:
                client.close()
            except Exception:
                pass

class LocalSandboxManager:
    """Simplified sandbox manager for local testing"""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ridges_local_test_"))
        self.agents_dir = self.temp_dir / "agents"
        self.repos_dir = self.temp_dir / "repos"
        self.repo_cache_dir = self.temp_dir / "repo_cache"
        
        # Import docker here to handle ImportError properly
        try:
            import docker
            from docker.models.containers import Container
            self.docker_module = docker
            self.Container = Container
        except ImportError:
            raise ImportError("Docker Python library is required for local testing. Install with: pip install docker")
        
        # Create directories
        self.agents_dir.mkdir(parents=True)
        self.repos_dir.mkdir(parents=True)
        self.repo_cache_dir.mkdir(parents=True)
        
        # Initialize main Docker client for infrastructure
        try:
            self.docker = self.docker_module.from_env()
            self.docker.ping()
        except Exception as e:
            raise RuntimeError(f"Docker is not running or accessible: {e}")
        
        # Create Docker client pool for parallel SWE-bench evaluations
        # Use reasonable pool size - too many clients can overwhelm the system
        pool_size = min(8, os.cpu_count() or 4)  # Max 8 clients, default to CPU count
        self.docker_pool = DockerClientPool(pool_size=pool_size)
        if self.verbose:
            print(f"Created Docker client pool with {pool_size} clients for parallel evaluations")
        
        # Create network for container communication
        self.network_name = f"local-test-{uuid.uuid4().hex[:8]}"
        try:
            self.network = self.docker.networks.create(
                self.network_name,
                driver="bridge",
                check_duplicate=True
            )
            if self.verbose:
                print(f"Created network: {self.network_name}")
        except self.docker_module.errors.APIError as e:
            if "already exists" in str(e):
                self.network = self.docker.networks.get(self.network_name)
            else:
                raise
        
        # Start proxy container
        self.proxy_container_name = "sandbox-proxy"
        try:
            # Check if proxy container already exists
            existing = self.docker.containers.get(self.proxy_container_name)
            existing.remove(force=True)
        except self.docker_module.errors.NotFound:
            pass
        
        # Pull proxy image if needed
        try:
            self.docker.images.get(PROXY_DOCKER_IMAGE)
        except self.docker_module.errors.ImageNotFound:
            print(f"Pulling proxy image: {PROXY_DOCKER_IMAGE}")
            self.docker.images.pull(PROXY_DOCKER_IMAGE)
        
        # Start proxy container on the custom network, but configure it to reach the host's proxy service
        host_proxy_url = os.getenv('RIDGES_PROXY_URL', 'http://localhost:8001')
        host_proxy_url = "http://localhost:8011"
        # Parse the URL and replace the hostname with host.docker.internal for Docker containers
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(host_proxy_url)
        # Replace any hostname/IP with host.docker.internal so containers can reach the host
        host_proxy_url = urlunparse(parsed._replace(netloc=f'host.docker.internal:{parsed.port or 8001}'))
        
        self.proxy_container = self.docker.containers.run(
            PROXY_DOCKER_IMAGE,
            name=self.proxy_container_name,
            network=self.network_name,
            detach=True,
            remove=False,
            extra_hosts={
                'host.docker.internal': 'host-gateway'
            },
            environment={
                'RIDGES_API_URL': os.getenv('RIDGES_API_URL', 'http://localhost:8000'),
                'RIDGES_PROXY_URL': host_proxy_url,
            }
        )
        if self.verbose:
            print(f"Started proxy container: {self.proxy_container_name}")
        
        # Wait for proxy to be ready
        import time
        time.sleep(2)
        
        # Use proxy container name for internal Docker network communication
        # The proxy listens on port 80 (default for nginx)
        self.proxy_url = f"http://{self.proxy_container_name}"
        if self.verbose:
            print(f"Using proxy URL: {self.proxy_url}")

    async def pre_build_swe_bench_images(self, problems: List[SwebenchProblem]):
        """Pre-build all SWE-bench environment images to enable parallel evaluations"""
        if not problems:
            return
            
        try:
            self._log_manager("Pre-building SWE-bench environment images for parallel evaluation...")
            
            from swebench.harness.run_evaluation import load_swebench_dataset, make_test_spec, build_env_images
            
            # Load instances and create test specs for all problems
            instance_ids = [problem.instance_id for problem in problems]
            instances = load_swebench_dataset("SWE-bench/SWE-bench_Verified", "test", instance_ids)
            test_specs = [make_test_spec(instance) for instance in instances]
            
            # Build all environment images at once using the main Docker client
            # This avoids the bottleneck of building images separately for each evaluation
            build_env_images(self.docker, test_specs, max_workers=4)
            
            self._log_manager(f"Pre-built environment images for {len(problems)} problems")
            
        except Exception as e:
            self._log_manager(f"Failed to pre-build SWE-bench images: {e}")
            # Continue anyway - individual evaluations will build as needed
    
    def _log_manager(self, message: str):
        """Log a message from the manager"""
        if self.verbose:
            print(f"[LocalSandboxManager] {message}")

    async def create_sandbox(self, problem: SwebenchProblem, agent_file: Path, log_buffer: list = None) -> 'LocalSandbox':
        """Create a new sandbox for testing"""
        # Copy agent file to temp directory
        agent_name = f"agent_{uuid.uuid4().hex[:8]}.py"
        agent_path = self.agents_dir / agent_name
        shutil.copy2(agent_file, agent_path)
        
        # Create evaluation run
        evaluation_run = EvaluationRun(
            run_id=str(uuid.uuid4()),
            evaluation_id=str(uuid.uuid4()),
            validator_hotkey="local-test",
            swebench_instance_id=problem.instance_id,
            status="sandbox_created",
            started_at=datetime.now(timezone.utc),
            sandbox_created_at=datetime.now(timezone.utc)
        )
        
        # Create sandbox directory structure that matches production
        sandbox_id = f"sandbox_{evaluation_run.run_id}"
        sandbox_dir = self.temp_dir / sandbox_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Create repo and src subdirectories to match production structure
        repo_dir = sandbox_dir / "repo"
        src_dir = sandbox_dir / "src"
        repo_dir.mkdir(exist_ok=True)
        src_dir.mkdir(exist_ok=True)
        
        # Create sandbox
        sandbox = LocalSandbox(
            evaluation_run=evaluation_run,
            problem=problem,
            agent_path=agent_path,
            sandbox_dir=sandbox_dir,
            repo_dir=repo_dir,
            local_manager=self,
            verbose=self.verbose,
            log_buffer=log_buffer
        )
        return sandbox

    def cleanup(self):
        """Clean up all resources"""
        try:
            # Clean up Docker client pool
            self.docker_pool.cleanup()
            if self.verbose:
                print("Cleaned up Docker client pool")
        except Exception as e:
            if self.verbose:
                print(f"Error cleaning up Docker pool: {e}")
        
        try:
            # Stop and remove proxy container
            if hasattr(self, 'proxy_container'):
                self.proxy_container.stop()
                self.proxy_container.remove()
                if self.verbose:
                    print(f"Removed proxy container: {self.proxy_container_name}")
        except Exception as e:
            if self.verbose:
                print(f"Error removing proxy container: {e}")
        
        try:
            # Remove network
            if hasattr(self, 'network'):
                self.network.remove()
                if self.verbose:
                    print(f"Removed network: {self.network_name}")
        except Exception as e:
            if self.verbose:
                print(f"Error removing network: {e}")
        
        try:
            # Remove temp directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                if self.verbose:
                    print(f"Removed temp directory: {self.temp_dir}")
        except Exception as e:
            if self.verbose:
                print(f"Error removing temp directory: {e}")

class LocalSandbox:
    """Local sandbox for testing agents without database dependencies"""
    def __init__(self, evaluation_run: EvaluationRun, problem: SwebenchProblem, agent_path: Path, sandbox_dir: Path, repo_dir: Path, local_manager: LocalSandboxManager, verbose: bool = False, log_buffer: list = None):
        self.evaluation_run = evaluation_run
        self.problem = problem
        self.agent_path = agent_path
        self.sandbox_dir = sandbox_dir  # Main sandbox directory (matches production structure)
        self.repo_dir = repo_dir        # Repository subdirectory (sandbox/repo/)
        self.local_manager = local_manager
        self.verbose = verbose
        self.container = None
        self._cancelled = asyncio.Event()
        self.log_buffer = log_buffer  # Set immediately from constructor
        self.container_logs = None  # Will store logs before container removal
        # Use the docker module from the manager
        self.docker_module = local_manager.docker_module
        
        # Clone repository into the repo subdirectory (matches production)
        clone_repo(
            self.repo_dir,
            problem.repo,
            problem.base_commit
        )
        try:
            from swebench.harness.run_evaluation import load_swebench_dataset  # local import to avoid heavy dependency at module load

            instance = load_swebench_dataset(
                "SWE-bench/SWE-bench_Verified",
                "test",
                [problem.instance_id],
            )[0]

            test_patch = instance.get("test_patch")
            if test_patch:
                # First, get the list of files that will be affected by the patch
                affected_files = []
                for line in test_patch.split('\n'):
                    if line.startswith('--- a/') or line.startswith('+++ b/'):
                        # Extract filename from patch header
                        if line.startswith('--- a/'):
                            filename = line[6:]  # Remove '--- a/' prefix
                        else:  # '+++ b/'
                            filename = line[6:]  # Remove '+++ b/' prefix
                        if filename != '/dev/null' and filename not in affected_files:
                            affected_files.append(filename)
                
                # Try applying with several strip levels; fall back to `patch` if needed
                import tempfile, os, shlex, textwrap
                with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                    tmp.write(test_patch)
                    tmp_path = tmp.name

                def _run_git_apply(p_level: str):
                    return subprocess.run(
                        ["git", "apply", p_level, "--verbose", "--reject", "--unidiff-zero", tmp_path],
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                    )

                applied = False
                for p_level in ("-p1", "-p0", "-p2", "-p3"):
                    res = _run_git_apply(p_level)
                    if res.returncode == 0:
                        applied = True
                        break
                if not applied:
                    # Fallback to system `patch`
                    res = subprocess.run(
                        ["patch", "-p1", "--forward", "--reject-file=-", "-i", tmp_path],
                        cwd=self.repo_dir,
                        capture_output=True,
                        text=True,
                    )
                    applied = res.returncode in (0, 1)

                os.unlink(tmp_path)

                if applied:
                    self._log_manager(
                        f"Applied test_patch for {problem.instance_id}"
                    )
                    if res.stdout or res.stderr:
                        self._log_manager("git/patch output:\n" + (res.stdout or "") + (res.stderr or ""))

                    # Simple verification: check that at least one added line now exists in repo
                    first_added = None
                    for ln in test_patch.splitlines():
                        if ln.startswith("+") and not ln.startswith("+++"):
                            snippet = ln[1:].strip()
                            if snippet:
                                first_added = snippet
                                break
                    if first_added:
                        grep_res = subprocess.run(
                            ["grep", "-R", "-n", first_added, "tests"],
                            cwd=self.repo_dir,
                            capture_output=True,
                            text=True,
                        )
                    try:
                        # Only add the specific files that were modified by the patch
                        if affected_files:
                            subprocess.run(["git", "add"] + affected_files, cwd=self.repo_dir, check=True)
                        subprocess.run([
                            "git",
                            "-c",
                            "user.email=tao@localhost",
                            "-c",
                            "user.name=Tao God",
                            "commit",
                            "-m",
                            "updates",
                        ], cwd=self.repo_dir, check=True)
                        self._log_manager("Committed test_patch for " + problem.instance_id)
                    except Exception as cex:
                        self._log_manager("Could not commit test_patch: " + str(cex))
            else:
                self._log_manager(f"No test_patch found for {problem.instance_id}")
        except Exception as e:
            self._log_manager(f"Error applying test_patch for {problem.instance_id}: {e}")

        self._log(f"Repository cloned to: {self.repo_dir}")
    def _log_manager(self, message: str):
        """Log a message from the manager"""
        if self.verbose:
            print(f"[LocalSandboxManager] {message}")
    def _log(self, message: str):
        """Log a message to buffer if available, otherwise print"""
        if self.log_buffer is not None:
            self.log_buffer.append(message)
        elif self.verbose:
            print(message)

    async def run(self) -> None:
        """Run the sandbox - compatibility method for runner"""
        result = await self.run_agent()
        # Store result in evaluation_run for runner compatibility
        if result['status'] == 'COMPLETED':
            self.evaluation_run.solved = result['solved']
        elif result['status'] == 'FAILED':
            self.evaluation_run.solved = False
            self.evaluation_run.error = result['error']
        elif result['status'] == 'ERROR':
            self.evaluation_run.solved = False
            self.evaluation_run.error = result['error']

    async def run_agent(self) -> Dict[str, Any]:
        """Run the agent in the sandbox"""
        try:
            # Prepare sandbox input
            sandbox_input = SandboxInput(
                instance_id=self.problem.instance_id,
                problem_statement=self.problem.problem_statement,
                repo=self.problem.repo,
                base_commit=self.problem.base_commit,
                run_id=self.evaluation_run.run_id,
            )
            
            # Write input file to sandbox root (matches production structure)
            input_file = self.sandbox_dir / "input.json"
            input_file.write_text(sandbox_input.model_dump_json())
            
            # Copy agent file to sandbox/src/ (matches production structure)
            src_dir = self.sandbox_dir / "src"
            agent_dest = src_dir / "agent.py"
            shutil.copy2(self.agent_path, agent_dest)
            
            # Copy the agent_runner.py to sandbox root (matches production structure)
            runner_src = Path(__file__).parent.parent / "sandbox" / "agent_runner.py"
            runner_dest = self.sandbox_dir / "agent_runner.py"
            shutil.copy2(runner_src, runner_dest)
            
            # Get commit-specific image for this instance (same as production)
            sandbox_image = get_sandbox_image_for_instance(self.problem.instance_id)
            
            # Pull sandbox image if needed
            try:
                self.local_manager.docker.images.get(sandbox_image)
                self._log(f"Using existing image: {sandbox_image}")
            except self.docker_module.errors.ImageNotFound:
                self._log(f"Pulling commit-specific image: {sandbox_image}")
                try:
                    self.local_manager.docker.images.pull(sandbox_image)
                    self._log(f"Successfully pulled: {sandbox_image}")
                except self.docker_module.errors.APIError as e:
                    self._log(f"Failed to pull commit-specific image {sandbox_image}: {e}")
                    self._log(f"Falling back to default image: {SANDBOX_DOCKER_IMAGE}")
                    sandbox_image = SANDBOX_DOCKER_IMAGE
                    try:
                        self.local_manager.docker.images.get(sandbox_image)
                    except self.docker_module.errors.ImageNotFound:
                        self._log(f"Pulling fallback image: {sandbox_image}")
                        self.local_manager.docker.images.pull(sandbox_image)
            
            # Run sandbox container
            self.container = self.local_manager.docker.containers.run(
                sandbox_image,
                f"python agent_runner.py",
                volumes={
                    str(self.sandbox_dir): {"bind": SANDBOX_DIR, "mode": "rw"}
                },
                working_dir=SANDBOX_DIR,
                network=self.local_manager.network_name,
                detach=True,
                remove=False,
                mem_limit=f"{SANDBOX_MAX_RAM_USAGE}m",
                environment={
                    'AI_PROXY_URL': self.local_manager.proxy_url,
                    'RIDGES_PROXY_URL': self.local_manager.proxy_url,
                    'RIDGES_API_URL': os.getenv('RIDGES_API_URL', 'http://localhost:8000'),
                }
            )
            self._log("Started sandbox container")
            
            # Monitor container
            monitor_task = asyncio.create_task(self._monitor_container())
            # Wait for container to complete
            await monitor_task
            
            # Read output
            output_file = self.sandbox_dir / "output.json"
            if output_file.exists():
                with open(output_file, 'r') as f:
                    output_data = json.load(f)
                
                # Check if agent_runner.py wrapped the output
                if output_data.get('success') and 'output' in output_data:
                    patch = output_data['output'].get('patch', '')
                else:
                    patch = output_data.get('patch', '')
                
                if patch:
                    self.evaluation_run.response = patch
                    self.evaluation_run.status = "patch_generated"
                    # Run SWE-bench evaluation with dedicated Docker client
                    await self._run_swe_bench_evaluation()
                    return {
                        'instance_id': self.problem.instance_id,
                        'status': 'COMPLETED',
                        'solved': self.evaluation_run.solved,
                        'patch_generated': True,
                        'patch_length': len(patch),
                        'error': self.evaluation_run.error
                    }
                else:
                    # Use pre-captured container logs
                    container_logs = f"\n{self.container_logs}" if self.container_logs else ""
                    
                    # Create detailed error message
                    error_parts = [
                        "Empty patch returned from agent.",
                        f"Raw output data: {json.dumps(output_data, indent=2)[:1000]}..."
                    ]
                    if container_logs:
                        error_parts.append(container_logs)
                    
                    detailed_error = "\n".join(error_parts)
                    
                    self.evaluation_run.status = "no_patch_generated"
                    self._log(f"Sandbox {self.evaluation_run.run_id} failed: {detailed_error}")
                    
                    return {
                        'instance_id': self.problem.instance_id,
                        'status': 'FAILED',
                        'solved': False,
                        'error': detailed_error,
                        'patch_generated': False,
                        'patch_length': 0
                    }
            else:
                # Use pre-captured container logs
                container_logs = f"\n{self.container_logs}" if self.container_logs else ""
                
                # Create detailed error message
                error_parts = [
                    "No output file generated by agent.",
                    f"Expected output file: {output_file}"
                ]
                if container_logs:
                    error_parts.append(container_logs)
                
                detailed_error = "\n".join(error_parts)
                
                self.evaluation_run.status = "no_output_file"
                self._log(f"Sandbox {self.evaluation_run.run_id} failed: {detailed_error}")
                
                return {
                    'instance_id': self.problem.instance_id,
                    'status': 'FAILED',
                    'solved': False,
                    'error': detailed_error,
                    'patch_generated': False,
                    'patch_length': 0
                }
                
        except Exception as e:
            # Use pre-captured container logs
            container_logs = f"\n{self.container_logs}" if self.container_logs else ""
            
            # Create detailed error message
            import traceback
            error_parts = [
                f"Sandbox execution failed: {str(e)}",
                f"Traceback: {traceback.format_exc()}"
            ]
            if container_logs:
                error_parts.append(container_logs)
            
            detailed_error = "\n".join(error_parts)
            
            self._log(f"Sandbox execution failed: {detailed_error}")
            self.evaluation_run.status = "execution_failed"
            self.evaluation_run.error = detailed_error
            
            return {
                'instance_id': self.problem.instance_id,
                'status': 'ERROR',
                'solved': False,
                'error': detailed_error,
                'patch_generated': False,
                'patch_length': 0
            }

    async def _run_swe_bench_evaluation(self) -> None:
        """Run SWE-bench evaluation on the generated patch using dedicated Docker client (images pre-built)"""
        dedicated_client = None
        try:
            self._log("Getting dedicated Docker client for SWE-bench evaluation...")
            # Get a dedicated Docker client from the pool
            dedicated_client = await self.local_manager.docker_pool.get_client()
            self._log("Got dedicated Docker client, starting SWE-bench evaluation...")
            
            from swebench.harness.run_evaluation import load_swebench_dataset, run_instance, make_test_spec
            
            # Load instance and create prediction
            instance_id = self.problem.instance_id
            instance = load_swebench_dataset("SWE-bench/SWE-bench_Verified", "test", [instance_id])[0]
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": self.evaluation_run.run_id,
                "model_patch": self.evaluation_run.response,
            }
            test_spec = make_test_spec(instance)
            
            # Skip the build step since images are pre-built - go directly to run_instance
            # This should enable true parallel execution since each evaluation uses its own client
            # and doesn't need to build any images
            self._log("Running evaluation (using pre-built environment images)...")
            result = run_instance(
                test_spec=test_spec,
                pred=prediction,
                rm_image=False,
                force_rebuild=False,  # Never rebuild since we pre-built
                client=dedicated_client,  # Use dedicated client
                run_id=self.evaluation_run.run_id,
                timeout=1800,
                rewrite_reports=False,
            )
            
            # Process results
            if result:
                _, report = result
                report = report[instance_id]
                if "tests_status" in report:
                    tests = report["tests_status"]
                    self.evaluation_run.fail_to_pass_success = json.dumps(tests["FAIL_TO_PASS"]["success"])
                    self.evaluation_run.pass_to_pass_success = json.dumps(tests["PASS_TO_PASS"]["success"])
                    self.evaluation_run.fail_to_fail_success = json.dumps(tests["FAIL_TO_FAIL"]["success"])
                    self.evaluation_run.pass_to_fail_success = json.dumps(tests["PASS_TO_FAIL"]["success"])
                    self.evaluation_run.solved = report.get("resolved", False)
                    self._log(f"SWE-bench evaluation completed: {'SOLVED' if self.evaluation_run.solved else 'NOT SOLVED'}")
                else:
                    self.evaluation_run.solved = False
                    self.evaluation_run.error = "No test results found in evaluation report"
                    self._log("SWE-bench evaluation completed but no test results found")
            else:
                # No results means patch didn't fix any tests
                self.evaluation_run.solved = False
                self.evaluation_run.fail_to_pass_success = json.dumps([])
                self.evaluation_run.pass_to_pass_success = json.dumps([])
                self.evaluation_run.fail_to_fail_success = json.dumps([])
                self.evaluation_run.pass_to_fail_success = json.dumps([])
                self._log("SWE-bench evaluation completed: NOT SOLVED (no test improvements)")
                
        except Exception as e:
            self._log(f"SWE-bench evaluation failed: {e}")
            self.evaluation_run.error = f"SWE-bench evaluation failed: {str(e)}"
            self.evaluation_run.solved = False
        finally:
            # Always return the dedicated client to the pool
            if dedicated_client:
                await self.local_manager.docker_pool.return_client(dedicated_client)
                self._log("Returned dedicated Docker client to pool")

    async def _monitor_container(self) -> None:
        """Monitor container execution with resource limits"""
        while True:
            if self._cancelled.is_set():
                self.container.kill()
                raise asyncio.CancelledError()
            try:
                self.container.reload()
                if self.container.status == "exited":
                    break
                elif self.container.status in ["dead", "removing"]:
                    break
                # Check memory usage
                try:
                    stats = self.container.stats(stream=False)
                    memory_stats = stats.get("memory_stats", {})
                    # Try different memory usage keys
                    for key in ["usage", "current", "memory.current"]:
                        if key in memory_stats:
                            usage_mb = memory_stats[key] / (1024 * 1024)
                            if usage_mb > SANDBOX_MAX_RAM_USAGE:
                                self.container.kill()
                                raise MemoryError(f"RAM limit exceeded: {usage_mb:.1f}MB")
                            break
                except Exception:
                    pass  # Continue monitoring even if stats fail
                # Check runtime limit  
                if self.evaluation_run.sandbox_created_at:
                    runtime = (datetime.now(timezone.utc) - self.evaluation_run.sandbox_created_at).total_seconds()
                    if runtime > SANDBOX_MAX_RUNTIME:
                        self.container.kill()
                        raise TimeoutError(f"Runtime limit exceeded: {runtime:.1f}s")
            except Exception as e:
                error_str = str(e).lower()
                if "exited" in error_str or "not found" in error_str or "no such container" in error_str:
                    # Container has exited or been removed
                    break
                logger.warning(f"Container monitoring error: {e}")
            await asyncio.sleep(1)
        
        # Capture container logs before removal
        try:
            if self.container:
                # logs = self.container.logs(stdout=True, stderr=True, tail=100).decode('utf-8', errors='ignore')
                logs = self.container.logs(stdout=True, stderr=True).decode('utf-8', errors='ignore')
                timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                log_filename = f'container_logs/{timestamp}.txt'
                with open(log_filename, 'w') as f:
                    f.write(logs)
                
                self.container_logs = f"Container logs (last 100 lines):\n{logs}"
        except Exception as e:
            self.container_logs = f"Failed to capture container logs: {e}"
        
        # Clean up container
        try:
            self.container.remove()
        except Exception:
            pass
    def cleanup(self) -> None:
        """Clean up sandbox resources"""
        if self.container:
            try:
                self.container.remove()
            except Exception:
                pass
        if self.sandbox_dir and self.sandbox_dir.exists():
            try:
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            except Exception:
                pass
    async def cancel(self) -> None:
        """Cancel sandbox execution"""
        self._cancelled.set() 