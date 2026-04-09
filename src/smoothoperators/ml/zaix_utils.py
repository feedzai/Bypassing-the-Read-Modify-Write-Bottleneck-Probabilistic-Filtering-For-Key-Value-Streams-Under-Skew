import zaix


class DataLoaderWrapper(LGBMDataLoader):

    def __init__(self, data_frame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_frame = data_frame

    def load_in_memory_dataset(self):
        data = lgbDataset(
            self.data_frame[self.feature_name],
            label=df[self.label_column],
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            reference=self.train_reference if self.train_reference else None,
            free_raw_data=self.free_raw_data,
            params=dataset_params,
        )

        if self.store_path:
            data.save_binary(self.store_path)

        return data
