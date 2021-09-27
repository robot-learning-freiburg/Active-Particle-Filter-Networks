#### Passive Localization

| Test Name |  for House3D env | for iGibson env|
|:---------:|:----------------:|:--------------:|
| Single apartment | rgb / rgbd / depth + gaussian/uniform + more/less particles | rgb / rgbd / depth / scan + gaussian/uniform + more/less particles |
| Multiple apartments | rgb / rgbd / depth + gaussian/uniform + more/less particles| rgb / rgbd / depth / scan + gaussian/uniform + more/less particles |

##### Tracking -> gaussian distribution without resampling for 25 episode steps
|  Mean Squared Error (cm)            |     RGB    |    Depth   |    RGB-D   |    Lidar   |
|:-----------------------------------:|:----------:|:----------:|:----------:|:----------:|
|  House3D (tracking)                 |   38.837   |   40.626   |   40.908   |   ------   |
|  iGibson (single aprt - obstacle)   |   18.168   |   43.144   |   18.097   |   43.356   |
|  iGibson (single aprt - floor_map)  |   ------   |   ------   |   18.404   |   ------   |
|  iGibson (15 aprt)                  |   ------   |   ------   |   53.195   |   ------   |
```
House3D RMSE(cm) for tracking - init_particles_distr 'tracking', init_particles_std '0.3' '0.523599', num_particles '300', map_pixel_in_meters '0.02', transition_std '0.' '0.', trajlen '24', resample 'false'
iGibson (single aprt) RMSE(cm) for tracking - init_particles_distr 'gaussian', init_particles_std '0.15' '0.523599', num_particles '300', map_pixel_in_meters '0.1', transition_std '0.' '0.', trajlen '24', resample 'false'
```

##### Global Localization -> uniform distribution without resampling for 100 episode steps

|         Success Rate (%)       |     RGB    |    Depth   |    RGB-D   |    Lidar   |
|:------------------------------:|:----------:|:----------:|:----------:|:----------:|
|  House3D (localization) - (1)  |   81.373   |   81.250   |   84.069   |   ------   |
|  House3D (localization) - (2)  |   87.990   |   84.681   |   88.971   |   ------   |
|  iGibson (single aprt)  - (1)  |   89.375   |   19.375   |   90.625   |   21.042   |
|  iGibson (single aprt)  - (2)  |   90.833   |   18.750   |   91.875   |   22.083   |
|  iGibson (single aprt)  - (3)  |   87.708   |    9.792   |   90.000   |    8.333   |
```
House3D success rate(%) for semi-global localization over one room (1)- init_particles_distr 'one-room', num_particles '500', map_pixel_in_meters '0.02', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
House3D success rate(%) for semi-global localization over one room (2)- init_particles_distr 'one-room', num_particles '1000', map_pixel_in_meters '0.02', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
iGibson (single aprt) success rate(%) for global localization (1)- init_particles_distr 'uniform', num_particles '500', map_pixel_in_meters '0.1', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
iGibson (single aprt) success rate(%) for global localization (2)- init_particles_distr 'uniform', num_particles '1000', map_pixel_in_meters '0.1', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
iGibson (single aprt) success rate(%) for global localization (3)- init_particles_distr 'uniform', num_particles '500', map_pixel_in_meters '0.1', transition_std '0.04' '0.0872665', trajlen '50', resample 'true', alpha_resample_ratio '1.0'
```

#### Active Localization

| Test Name | for iGibson env|
|:---------:|:--------------:|
| Single apartment (fixed pose) | belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |
| Single apartment (random pose) | belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |
| Multiple apartments (random pose) |belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |

![#BB8E01](https://via.placeholder.com/15/BB8E01/000000?text=+) sampling random pose from 0.5 x 0.5 meters area with episode length 25\
![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) sampling random pose from 1.0 x 1.0 meters area with episode length 25\
![#B90054](https://via.placeholder.com/15/B90054/000000?text=+) sampling random pose from 1.0 x 1.0 meters area with episode length 50\
![#B3B3B3](https://via.placeholder.com/15/B3B3B3/000000?text=+) sampling random pose from 2.0 x 2.0 meters area with episode length 25

![Screenshot from 2021-09-20 12-10-54](https://user-images.githubusercontent.com/4303534/133989025-7fde1bb4-6a5e-4d60-b07e-8aaea35f17a4.png "Rl agent")
