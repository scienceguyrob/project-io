# Building and Running project-io Locally

This guide walks through building the project-io Docker image from source and running it locally with JupyterLab. It covers macOS, Windows, and Linux.

---

## Prerequisites

You need Docker installed and running on your machine.

### macOS and Windows

Install **Docker Desktop**:

1. Go to [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Download the version for your operating system
   - **macOS:** choose Apple Silicon (M1/M2/M3/M4) or Intel depending on your chip. If you are unsure, click the Apple menu → About This Mac — it will say either "Apple M..." or "Intel"
   - **Windows:** download the Windows installer
3. Run the installer and follow the prompts
4. Launch Docker Desktop and wait until the status shows **Docker Desktop is running**

### Linux

Install Docker Engine directly — Docker Desktop is optional on Linux:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Log out and back in for the group change to take effect.

### Verify Docker is working

Open a terminal (macOS/Linux) or Command Prompt / PowerShell (Windows) and run:

```bash
docker --version
```

You should see something like `Docker version 27.x.x`. If you get `command not found` or an error about the daemon not running, Docker is not installed or not started.

---

## Step 1 — Get the project

Clone the repository:

```bash
git clone https://github.com/scienceguyrob/project-io.git
```

Then move into the project folder:

```bash
cd project-io
```

All subsequent commands in this guide are run from this folder.

If you do not have Git installed, you can instead download the repository as a ZIP from GitHub (click **Code → Download ZIP**), unzip it, and open a terminal in the unzipped folder.

---

## Step 2 — Check the file structure

Before building, confirm the key files are in place:

```bash
ls
```

You should see `index.html`, the `web/` folder, `docker/`, and the `Notebook_0/` through `Notebook_13/` folders. The Dockerfile expects all of these to be present at the project root.

---

## Step 3 — Build the image

Run the following command from the project root (don't forget the period at the end!):

```bash
docker build -f docker/Dockerfile -t project-io .
```

Breaking that down:

| Part | What it does |
|---|---|
| `docker build` | Tells Docker to build an image |
| `-f docker/Dockerfile` | Points to the Dockerfile inside the `docker/` folder |
| `-t project-io` | Names (tags) the resulting image `project-io` |
| `.` | Sets the build context to the current folder — Docker uses this to find all the project files to copy in |

**The first build will take several minutes.** Docker is downloading the base Python image and installing all the Python dependencies (numpy, pandas, scikit-learn, JupyterLab, and others). A lot of output will scroll past — this is normal.

Subsequent builds are much faster because Docker caches each layer. As long as you have not changed the `pip install` block in the Dockerfile, the dependencies will not be reinstalled.

When the build finishes successfully you will see something like:

```
Successfully built a1b2c3d4e5f6
Successfully tagged project-io:latest
```

---

## Step 4 — Confirm the image exists

```bash
docker images
```

You should see `project-io` listed with a `latest` tag and a size around 2–3 GB.

---

## Step 5 — Run the container

```bash
docker run --rm -p 8888:8888 project-io
```

Breaking that down:

| Part | What it does |
|---|---|
| `docker run` | Starts a container from the image |
| `--rm` | Automatically removes the container when you stop it, keeping things tidy |
| `-p 8888:8888` | Maps port 8888 on your machine to port 8888 inside the container |
| `project-io` | The image to run |

Leave this terminal window open — closing it stops the container.

You will see JupyterLab output like:

```
[I ServerApp] Serving notebooks from local directory: /home/notebook
[I ServerApp] Jupyter Server is running at:
[I ServerApp] http://localhost:8888/lab
```

---

## Step 6 — Open JupyterLab

Open your browser and go to:

```
http://localhost:8888
```

JupyterLab will open with the file browser showing all the project notebooks and folders. No password or token is required.

To open a specific HTML page directly in the browser (recommended over viewing it inside JupyterLab, as some interactive features work better this way):

```
http://localhost:8888/files/index.html
http://localhost:8888/files/web/search.html
http://localhost:8888/files/web/challenges.html
http://localhost:8888/files/web/data.html
http://localhost:8888/files/web/how_to.html
```

---

## Step 7 — Stop the container

Go back to the terminal window where the container is running and press:

```
Ctrl + C
```

The container will shut down cleanly. Because you used `--rm`, it is automatically deleted. The image itself is preserved and you can run it again any time with the same `docker run` command.

---

## Saving your work between sessions

By default, any changes you make to notebooks inside the container are lost when the container stops — the image itself is read-only.

To save work between sessions, mount a local folder into the container using the `-v` flag. The syntax differs slightly by operating system.

### macOS and Linux

```bash
docker run --rm -p 8888:8888 \
  -v /path/to/your/work:/home/notebook/work \
  project-io
```

For example, if you want to use a folder called `project-io-work` on your desktop:

```bash
mkdir -p ~/Desktop/project-io-work

docker run --rm -p 8888:8888 \
  -v ~/Desktop/project-io-work:/home/notebook/work \
  project-io
```

### Windows (Command Prompt)

```cmd
docker run --rm -p 8888:8888 -v C:\Users\YourName\project-io-work:/home/notebook/work project-io
```

### Windows (PowerShell)

```powershell
docker run --rm -p 8888:8888 -v C:/Users/YourName/project-io-work:/home/notebook/work project-io
```

In all cases, replace the local path with a folder that exists on your machine. A `work/` directory will appear in the JupyterLab file browser. Anything you save there will be written to your local folder immediately and will persist after the container stops.

> **Windows note:** Docker on Windows requires the drive you are sharing (e.g. `C:`) to be enabled for file sharing in Docker Desktop settings. If you get a permission error on the volume mount, check **Docker Desktop → Settings → Resources → File Sharing**.

---

## Rebuilding after changes

If you add or modify notebooks or other project files and want those changes reflected in the image, rebuild it:

```bash
docker build -f docker/Dockerfile -t project-io .
```

If you only changed notebook or web content (not the `pip install` dependencies), Docker will reuse the cached dependency layers and only re-copy the changed files, so the rebuild will be fast.

---

## Troubleshooting

### Port 8888 is already in use

Another process (perhaps a locally running JupyterLab) is using port 8888. Either stop that process, or map the container to a different port on your machine:

```bash
docker run --rm -p 8889:8888 project-io
```

Then open `http://localhost:8889` instead.

---

### `docker: command not found`

Docker is not installed, or on macOS/Windows, Docker Desktop is not running. Start Docker Desktop and wait for it to fully initialise before trying again.

---

### `Cannot connect to the Docker daemon`

Docker Desktop is not running. Open it from your applications and wait for the status indicator to show it is ready.

---

### The build fails with a pip error

This usually means a pinned package version is no longer available from PyPI. The build output will identify which package failed. Open `docker/Dockerfile`, update the version number for that package in the `pip install` block, and rebuild.

---

### JupyterLab opens but a notebook throws an import error

A dependency used by that notebook is not in the `pip install` list in the Dockerfile. Add it (with a pinned version) and rebuild.

---

### Volume mount produces a permission error on Linux

On Linux, the container runs as a user with UID 1000. If the local folder you are mounting is owned by a different user, Docker may not be able to write to it. Fix it by setting ownership on the local folder:

```bash
sudo chown -R 1000:1000 /path/to/your/work
```

---

## Pulling from Docker Hub (alternative to building)

Once the image has been published to Docker Hub you will be able to skip the build entirely and just pull and run it directly:

```bash
docker run --rm -p 8888:8888 scienceguyrob/project-io
```

Docker will download the image automatically on first run. Check the [project-io GitHub repository](https://github.com/scienceguyrob/project-io) for the latest status on Docker Hub availability.
