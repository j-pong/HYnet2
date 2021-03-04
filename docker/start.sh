#!/bin/bash

echo "[start.sh] executed"

if [ -z "${PUID}" -o -z "${PGID}" ]; then
    exec /bin/bash
else
    if [ "${PUID}" -eq 0 -o "${PGID}" -eq 0 ]; then
        echo "[start.sh] Nothing to do here." ; exit 0
    fi
fi

USER=jpong
PGID=${PGID:-5555}
PUID=${PUID:-5555}
echo "PUID=${PUID}"
echo "PGID=${PGID}"

groupmod -o -g "$PGID" ${USER}
usermod -o -u "$PUID" ${USER}
chown -R ${PUID}:${PGID} /home/${USER}

su - jpong -c "ssh-keygen -q -t rsa -b 4096 -f ~/.ssh/id_rsa -C jpong[${PUID}-${PGID}]@$(hostname) -N ''"
echo "[start.sh] ssh-key generated."

#-------------------------------------------------------------------------------
exec /usr/sbin/sshd -D