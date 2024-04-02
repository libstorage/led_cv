
## Visual verification of storage LED state

To really make sure that what the hardware is telling us and what a human actually sees, we added a USB attached
camera and code to interpret video captures.

So that we can take this:

![example](https://github.com/libstorage/led_cv/assets/2520480/dbe470b9-cdad-4b8e-9496-74435f81d979)

and end up with:

```yaml
results:
- state: 4
  wwn: 50014ee26049cc9d
- state: 2
  wwn: 50014ee20aeb1530
- state: 4
  wwn: 50014ee20ae9bafa
- state: 2
  wwn: 50014ee2b59fb711
statekey: '{OFF: 1, NORM: 2, LOCATE: 3, FAULT: 4, LOCATE_FAULT: 5, UNKNOWN: 6}'
```

It's a bit more complicated, as some of the LED states are blinking, so we need to take a number of samples to
determine what is going on.  See source code for all the details.

**Caveats: This code is very specific to one of our test systems and would need changes for others to utilize.**
