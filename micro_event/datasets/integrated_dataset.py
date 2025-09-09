import numpy as np
import torch
from .dataset_mass import SleepEventDataset
from .dataset_hn import SleepEventDatasetEBX

class IntegratedDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 mass_subject_ids, honey_subject_ids,
                 augmented_page=False,
                 border_size=2.6, normalize_clip=True, pages_subset="N2",
                 normalization_mode="N2", event_type='spindle', annotator='E1',
                 page_duration=20, target_fs=200):
        self.event_type          = event_type.lower()
        self.annotator           = annotator.upper()
        self.fs                  = target_fs      
        self.subject_data        = {}
        self.page_subset         = pages_subset  # 'N2' or 'all'
        self.normalization_mode  = normalization_mode
        self.normalize_clip      = normalize_clip
        self.page_duration       = page_duration  # in seconds
        self.page_size           = int(self.fs * self.page_duration)  # in samples
        self.unknown_id          = "?"
        self.n2_id               = "2"  # N2 stage ID
        self.min_kc_duration     = 0.2
        self.aligned_downsample  = True 
        self.augmented_page      = augmented_page
        self.border_size         = int(np.round(border_size * self.fs))  # Convert to samples
        self.stride              = 8 
        
        mass_dir = "/home/honeynaps/data/MASS_FILES/SS2"
        honey_dir = "/home/honeynaps/data/HN_DATA_MW"

        honey_dataset = SleepEventDatasetEBX(honey_dir, honey_subject_ids,
                                             event_type=event_type,
                                             pages_subset=pages_subset,
                                             augmented_page=augmented_page,
                                             page_duration=page_duration,
                                             target_fs=target_fs)
        signal2, marks2, page_masks2 = honey_dataset.signals, honey_dataset.marks, honey_dataset.page_masks
        if len(mass_subject_ids) != 0:
            mass_dataset = SleepEventDataset(mass_dir, mass_subject_ids,
                                        event_type=event_type,
                                        annotator=annotator,
                                        norm_mad=True,
                                        pages_subset=pages_subset,
                                        normalization_mode=normalization_mode,
                                        augmented_page=augmented_page,
                                        page_duration=page_duration,
                                        target_fs=target_fs)
            signal1, marks1, page_masks1 = mass_dataset.signals, mass_dataset.marks, mass_dataset.page_masks
            self.signals = np.concatenate((signal1, signal2), axis=0)  # shape (total_segments, total_length)
            self.marks   = np.concatenate((marks1, marks2), axis=0)    # shape (total_segments, total_length)
            self.page_masks = np.concatenate((page_masks1, page_masks2), axis=0)  # shape (total_segments, total_length)
        else:
            self.signals = signal2
            self.marks   = marks2
            self.page_masks = page_masks2
        
    def __len__(self):
        return self.signals.shape[0]
    
    def __getitem__(self, idx):
        feat  = self.signals[idx]    # shape (total_length,)
        label = self.marks[idx]      # shape (total_length,)
        mask  = self.page_masks[idx] # shape (total_length,)

        total_length = feat.shape[-1]
        crop_length = self.page_size + 2 * self.border_size
        if total_length > crop_length:
            # Choose a random start index for cropping
            max_offset = total_length - crop_length
            start = np.random.randint(0, max_offset + 1)
            end = start + crop_length
            feat = feat[start:end]
            label = label[start:end]
            mask = mask[start:end]

        # 2. **Remove border** regions from label and mask to get the central page region
        center_label = label[self.border_size : -self.border_size]   # shape = page_size
        center_mask  = mask[self.border_size : -self.border_size]    # shape = page_size

        # 3. **Downsample** the label and mask by the stride factor to match model output rate
        if self.aligned_downsample:
            block_size = self.stride

            trimmed_length = (len(center_label) // block_size) * block_size
            label_blocks = center_label[:trimmed_length].reshape(-1, block_size)
            mask_blocks  = center_mask[:trimmed_length].reshape(-1, block_size)

            label_down = label_blocks.mean(axis=1)
            mask_down  = mask_blocks.mean(axis=1)

            label_down = np.rint(label_down).astype(np.float32)
            mask_down = np.rint(mask_down).astype(np.float32)
        else:
            label_down = center_label[:: self.stride].astype(np.float32)
            mask_down  = center_mask[:: self.stride].astype(np.float32)

        feat_tensor = torch.from_numpy(feat).float()
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(0)  # shape (1, length) 

        label_tensor = torch.from_numpy(label_down).float()  # shape (downsampled_length,)
        mask_tensor  = torch.from_numpy(mask_down).float()   # shape (downsampled_length,)
        return feat_tensor, label_tensor, mask_tensor
        

if __name__ == "__main__":
    # Example usage
    data_dir = "/home/honeynaps/data/MASS_FILES/SS2"
    subject_ids = [1, 2, 3, 4]  # Example subject IDs
    dataset = IntegratedDataset(data_dir, subject_ids, event_type='kcomplex', annotator='E1')
    
    # Prepare the data
    x, y, page_mask = dataset._prepare_data()
    
    print("Signals shape:", x.shape)
    print("States shape:", y.shape)
    print("Page mask shape:", page_mask.shape)
