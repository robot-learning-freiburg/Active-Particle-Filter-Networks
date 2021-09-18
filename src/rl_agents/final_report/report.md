#### Passive Localization

| Test Name |  for House3D env | for iGibson env|
|:---------:|:----------------:|:--------------:|
| Single apartment | rgb / rgbd / depth + gaussian/uniform + more/less particles | rgb / rgbd / depth / scan + gaussian/uniform + more/less particles |
| Multiple apartments | rgb / rgbd / depth + gaussian/uniform + more/less particles| rgb / rgbd / depth / scan + gaussian/uniform + more/less particles |

#### Active Localization

| Test Name | for iGibson env|
|:---------:|:--------------:|
| Single apartment (fixed pose) | belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |
| Single apartment (random pose) | belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |
| Multiple apartments (random pose) |belief map / k means cluster + rgb / rgbd / depth + gaussian/uniform + more/less particles |


|                          |     RGB    |    Depth   |    RGB-D   |    Lidar   |
|:------------------------:|:----------:|:----------:|:----------:|:----------:|
|  House3D (tracking)      |   38.837   |   40.626   |   40.908   |   ------   |
|  iGibson (single aprt)   |   18.168   |   43.144   |   18.097   |   43.356   |
```
House3D RMSE(cm) for tracking - init_particles_distr 'tracking', init_particles_std '0.3' '0.523599', num_particles '300', map_pixel_in_meters '0.02', transition_std '0.' '0.', trajlen '24', resample 'false'
iGibson (single aprt) RMSE(cm) for tracking - init_particles_distr 'gaussian', init_particles_std '0.15' '0.523599', num_particles '300', map_pixel_in_meters '0.1', transition_std '0.' '0.', trajlen '24', resample 'false'
```

|                                |     RGB    |    Depth   |    RGB-D   |    Lidar   |
|:------------------------------:|:----------:|:----------:|:----------:|:----------:|
|  House3D (localization) - (1)  |   81.373   |   81.250   |   84.069   |   ------   |
|  House3D (localization) - (2)  |   87.990   |   84.681   |   88.971   |   ------   |
|  iGibson (single aprt)  - (1)  |   89.375   |   19.375   |   90.625   |   21.042   |
|  iGibson (single aprt)  - (2)  |   90.833   |   18.750   |   91.875   |   22.083   |
```
House3D success rate(%) for semi-global localization over one room (1)- init_particles_distr 'one-room', num_particles '500', map_pixel_in_meters '0.02', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
House3D success rate(%) for semi-global localization over one room (2)- init_particles_distr 'one-room', num_particles '1000', map_pixel_in_meters '0.02', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
iGibson (single aprt) success rate(%) for global localization (1)- init_particles_distr 'uniform', num_particles '500', map_pixel_in_meters '0.1', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
iGibson (single aprt) success rate(%) for global localization (2)- init_particles_distr 'uniform', num_particles '1000', map_pixel_in_meters '0.1', transition_std '0.04' '0.0872665', trajlen '100', resample 'true', alpha_resample_ratio '1.0'
```
