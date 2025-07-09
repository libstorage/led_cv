# Visual verification of storage LED state

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

## What to do when the camera gets moved

1. Collect an updated image, figure out what LEDs you have access too and update the `REG_0 - REG_N` variables
2. Update `data.learn`, which is done by the following
   * Manually set all the LEDs of interest to normal, then run `./led_determine.py collect G G G G > data.learn`
   * Manually set all the LEDs of interest to failure, then run `./led_determine.py collect R R R R >> data.learn`
3. Test it
4. Update the `config.yaml` to ensure that each LED has the correct WWN
