---
process: sample-stock
department: lab-operations
---

# Checking Sample Stock

Sample inventory is managed in the Sample Stock application. This guide
explains how to check current quantities, see what is reserved, and
request a restock.

## Locating a Sample

Open the Sample Stock app and search by sample identifier, project code,
or storage location. The default view shows samples assigned to your
team. Toggle "All samples" to see organization-wide inventory, subject to
your access level.

## Reading the Quantity Display

Each sample row shows three quantities:

- On hand: the number of physical units currently in storage.
- Reserved: units allocated to an upcoming protocol or experiment.
- Available: on hand minus reserved.

Always plan against the available quantity, not the on-hand quantity. A
sample with five units on hand and four reserved leaves only one usable
unit.

## Reserving Stock

To reserve stock for a planned protocol, click "Reserve" on the sample
row and enter the quantity, the protocol identifier, and the planned
date. Reservations expire automatically forty-eight hours after the
planned date if not consumed; this keeps the available quantity honest.

## Requesting a Restock

When the available quantity falls below the per-sample threshold, the
sample is flagged in red. Click "Request restock" to open a restock
request. The request routes to logistics, who fulfil it from a central
repository or from an external supplier. Do not open a ServiceNow
incident for restocks — the incident channel is for tool problems, not
stock requests.

## Discrepancies

If the displayed quantity does not match what you physically count,
report the discrepancy through the assistant or directly in the Sample
Stock app under "Report discrepancy". Accurate counts are critical
because downstream booking decisions depend on them.
