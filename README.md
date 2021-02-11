# GGF_realtime_forecast
Program GGF_RTF.py forecasts ground geomagnetic field in real time, for about an hour into the future.

The program GGF_RTF.py makes a real-time forecast of the external magnetic field perturbation (i.e. 
 excluding the internal field contribution) at the three British Geological 
 Survey magnetometer stations: Eskdalemuir, Hartland and Lerwick.
 
 Output file column format is as follows.
  Day-Month-Year    Hour:Minute    [Model Type Flag]    Esk x    Esk y    Esk z    Had x    Had y    Had z    Ler x    Ler y    Ler z

 Where: 
 'Esk' is the observatory Eskdalemuir, 'Har' is Hartland, 'Ler' is 
  Lerwick, and 'x','y','z' are the ground geomagnetic field perturbation 
  components.
 The Model Type Flag has the following meanings.
  0: no problems: model forecast is based on the epsilon coupling function.
  1: plasma data were not available for this epoch of the real-time soalr wind 
      stream, so the forcast is based on just the magnetometer data, using the
      coupling function (IMF_magnitude^3) * (sin(clock_angle/2)^4).
  2: no solar wind data were available, so the output is nan.

 User notes.
 -- Don't compute dB/dt when there is a flag change.
 -- GGF baselines are preliminary values, and will change.
 
author: Robert Shore: robore@bas.ac.uk
