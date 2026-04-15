from smoothoperators import feature_profiles
from smoothoperators import event_filtering

if __name__ == '__main__':
    PATH_TO_WIKIPEDIA_CLICKHOUSE = 'YOUR PATH TO WIKIPEDIA CLICKHOUSE'
    event_filter = event_filtering.EventFilter(
        'features',
        'events',
        'join_key',
        'unix_seconds',
        'event_time_unix',
    )
    event_filter.thresholds = [
        0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.002, 0.003
    ]
    event_filter.half_lifes = ['day']
    event_filter.num_trials = 1
    entity = 'editor_id'

    kept_volume_day_ds = {
        f'thresholds_{t}': []
        for t in event_filter.thresholds
    }

    kept_volume_day_ds_vr = {
        f'thresholds_{t}': []
        for t in event_filter.thresholds
    }

    kept_volume_day_vanilla = {
        f'thresholds_{t}': []
        for t in event_filter.thresholds
    }
    with feature_profiles.ClickHouseConnector(PATH_TO_WIKIPEDIA_CLICKHOUSE):
        for i in range(15):
            try:
                volumes = event_filter.kept_volume(
                    entity, i, double_stochastic=True).loc[0].to_dict()
                for t in event_filter.thresholds:
                    kept_volume_day_ds[f'thresholds_{t}'].append(volumes[
                        f'kept_volume_half_life_day_threshold_{event_filtering.custom_sci(t)}']
                                                                 )
            except Exception as e:
                print(f'Failed to fetch trial {i} double stochastic')

        for i in range(15):
            try:
                volumes = event_filter.kept_volume(
                    entity, i, double_stochastic=True,
                    var_alpha=1).loc[0].to_dict()
                for t in event_filter.thresholds:
                    kept_volume_day_ds_vr[f'thresholds_{t}'].append(volumes[
                        f'kept_volume_half_life_day_threshold_{event_filtering.custom_sci(t)}']
                                                                    )
            except Exception as e:
                print(
                    f'Failed to fetch trial {i} double stochastic with var reduction'
                )
                print(e)

        for i in range(0, 15):
            try:
                volumes = event_filter.kept_volume(
                    entity, i, double_stochastic=False).loc[0].to_dict()
                for t in event_filter.thresholds:
                    kept_volume_day_vanilla[f'thresholds_{t}'].append(volumes[
                        f'kept_volume_half_life_day_threshold_{event_filtering.custom_sci(t)}']
                                                                      )
            except Exception as e:
                print(f'Failed to fetch trial {i} vanilla')

    print('\nDOUBLE STOCHASTIC')
    print(kept_volume_day_ds)
    print('\nDOUBLE STOCHASTIC VARIANCE REDUCTION')
    print(kept_volume_day_ds_vr)
    print('\nVANILLA')
    print(kept_volume_day_vanilla)
