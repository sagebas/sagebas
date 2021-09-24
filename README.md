# GGF_realtime_forecast
Program GGF_RTF.py forecasts ground geomagnetic field in real time, for about an hour into the future.

The program GGF_RTF.py makes a real-time forecast of the external magnetic field perturbation (i.e. 
 excluding the internal field contribution) at the three British Geological 
 Survey magnetometer stations: Eskdalemuir, Hartland and Lerwick.
 
 Output file column format is as follows.
  Day-Month-Year    Hour:Minute    [Model Type Flag]    Substorm onset probability    Esk x    Esk y    Esk z    Had x    Had y    Had z    Ler x    Ler y    Ler z

 Where: 
 'Esk' is the observatory Eskdalemuir, 'Har' is Hartland, 'Ler' is 
  Lerwick, and 'x','y','z' are the ground geomagnetic field perturbation 
  components.
 The substorm onset probability defines the onset likelihood (from 0 to 1, 
  with 1 the most likely), for the hour following the model run time. Hence, 
  the onset forecast is only valid for that hour: the output file has nan 
  values for onset probabilities at forecast epochs greater than [current time
  + 60 mins].
 The Model Type Flag has the following meanings.
  0: no problems: model forecast is based on the epsilon coupling function.
  1: plasma data were not available for this epoch of the real-time solar wind 
      stream, so the forcast is based on just the magnetometer data, using the
      coupling function (IMF_magnitude^3) * (sin(clock_angle/2)^4).
  2: no solar wind data were available, so the output is nan.

 User notes.
 -- Don't compute dB/dt when there is a flag change.
 -- GGF baselines are preliminary values, and will change.
 
 The Dockerfile can be run using the following commands:
docker builder prune -a (to remove previous cloned github repositories and make room for newer ones)
docker build -f [on the local machine, the directory and filename of the Dockerfile you sourced from the SPIDER github repo] -t ggf_image . (to build the image)
docker run --name ggf_container ggf_image GGF_RTF.py (to run the Python script ‘GGF_RTF.py’ from the image in a container)
docker cp ggf_container:/root/run_environment/Temp_storage_for_output_GGF_forecast/ [on the local machine, the directory where you wish the output of the Python script to be copied to] (this copies the output forecast from a location in the Docker container to somewhere else on the local machine)

@author: Robert Shore: robore@bas.ac.uk
