# Virtual Session Troubleshooting

Many laboratory workflows depend on virtual sessions: remote desktops
that host instrument software, analysis tools, and shared data. This
document covers the most common problems and how to resolve them.

## Logging In

Sessions are accessed through the virtual login device that sits next to
your workstation. Insert your credential, unlock the device with your
PIN, and choose the session from the list. The first session of the day
may take up to two minutes to start; subsequent reconnects are usually
under thirty seconds.

If the device does not recognise your credential, try a different USB
port. If the issue persists, open a ServiceNow incident — do not request
a replacement device by email.

## Session Disconnects

Brief disconnects of a few seconds are normal when moving between
laboratory devices, because the session follows your credential rather
than the physical machine. If a disconnect lasts more than thirty
seconds, reconnect manually from the device. Repeated disconnects within
the same hour indicate a network problem and warrant an incident.

## Application Refuses to Launch Inside a Session

If an internal application fails to launch inside the virtual session
but launches on your local machine, the underlying cause is usually a
missing permission inside the session profile, not a bug in the
application. Open a ServiceNow incident referencing both the application
name and the session identifier shown in the bottom-right of the desktop.

## Performance Problems

Slowness inside a virtual session is most often caused by an overloaded
shared host. Switching to a different session host — visible in the
session menu — usually resolves the immediate problem. If the issue
recurs on multiple hosts during the same shift, open an incident.

## Saving Work

Files inside a virtual session live on a shared network volume. They
persist across reconnects but are subject to the standard backup
schedule. For analyses you cannot afford to lose, export a copy to your
team's dedicated share at the end of the working day.
