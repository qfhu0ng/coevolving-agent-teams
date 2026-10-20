#!/bin/sh
# Optional project-local Docker executable for the authorized Linux server.
# Do not install system-wide or alter the user's Docker group membership.
# Keep the Docker client's UID: docker cp creates host files as the client user.
# A root client would leave root-owned temporary evaluator XML files that the
# non-root Python evaluator cannot chmod. The docker group is scoped to this
# child process; this does not add the account to a persistent system group.
programbench_client_user=$(id -un)
exec sudo -n -u "$programbench_client_user" -g docker /usr/bin/docker "$@"
