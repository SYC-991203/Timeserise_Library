import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pandas as pd 
def visualize_attention(attention_array, save_dir, file_prefix):
    """
    Visualize and save feature-to-feature attention as heatmaps with normalization.

    Args:
        attention_data (tuple): A tuple containing tensors with attention weights.
                                Each tensor has the shape (batch_size, num_heads, num_vars, model_dim).
        save_dir (str): Directory where the visualizations will be saved.
        file_prefix (str): Prefix for the saved image filenames (default: 'attention').

    Returns:
        None
    """


    # Step 1: 对头取平均，self_Attention会得到（total_len，input_len，input_len）
    attention_mean_heads = np.mean(attention_array, axis=1)  # Shape: (batch_size, num_vars, model_dim)

    # Step 2: Compute feature-to-feature attention for each batch
    # Here, we compute dot products between features to get feature-to-feature attention
    batch_size, num_vars, model_dim = attention_mean_heads.shape
    batch_feature_attention = np.zeros((batch_size, num_vars, num_vars))

    for b in range(batch_size):
        for i in range(num_vars):
            for j in range(num_vars):
                batch_feature_attention[b, i, j] = np.dot(
                    attention_mean_heads[b, i], attention_mean_heads[b, j]
                )

    # Step 3: Average across all batches to get a single feature-to-feature attention matrix
    global_feature_attention = np.mean(batch_feature_attention, axis=0)  # Shape: (num_vars, num_vars)

    # Step 4: Normalize the feature-to-feature attention matrix
    min_value = global_feature_attention.min()
    max_value = global_feature_attention.max()
    normalized_attention = (global_feature_attention - min_value) / (max_value - min_value)

    # Step 5: Visualize the normalized feature-to-feature attention matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(normalized_attention, cmap='plasma', interpolation='nearest')
    plt.title(f"Normalized Feature-to-Feature Attention")
    plt.xlabel("Features")
    plt.ylabel("Features")

    # Define feature names
    feature_names = [
    r"$\eta$", r"$\phi$", r"$\rho$", r"$\sigma$", r"$\alpha$",
    r"$\eta_{t}$", r"$\eta_{s}$", r"$\eta_{r}$",
    r"$\phi_{t}$", r"$\phi_{s}$", r"$\phi_{r}$",
    r"$\rho_{t}$", r"$\rho_{s}$", r"$\rho_{r}$",
    r"$\sigma_{t}$", r"$\sigma_{s}$", r"$\sigma_{r}$",
    r"$\alpha_{t}$", r"$\alpha_{s}$", r"$\alpha_{r}$"
    ]


    # Ensure the number of features matches the names
    if len(feature_names) != num_vars:
        raise ValueError("Number of feature names does not match num_vars.")

    plt.xticks(range(num_vars), labels=feature_names, rotation=90)
    plt.yticks(range(num_vars), labels=feature_names)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label("Attention Weight")

    plt.tight_layout()

    # Save the figure
    filename = f"LCCH_{file_prefix}_feature_attention.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved normalized feature-to-feature attention heatmap: {save_path}")



def visualize_self_attention(attention_array, save_dir, file_prefix):
    """
    Visualize and save feature-to-feature attention as heatmaps with normalization.

    Args:
        attention_data (tuple): A tuple containing tensors with attention weights.
                                Each tensor has the shape (batch_size, num_heads, num_vars, model_dim).
        save_dir (str): Directory where the visualizations will be saved.
        file_prefix (str): Prefix for the saved image filenames (default: 'attention').

    Returns:
        None
    """


    # Step 1: 对头取平均，self_Attention会得到（total_len，input_len，input_len）
    attention_mean_heads = np.mean(attention_array, axis=1) 
    attention_weights = attention_mean_heads[-1,:,:]
    pd.DataFrame(attention_weights).to_csv(f"{file_prefix}_attention.csv", index=False, header=False) # Shape: (batch_size, num_vars, model_dim)
    print("Saved the data")
    # Step 2: Compute feature-to-feature attention for each batch
    # Here, we compute dot products between features to get feature-to-feature attention
    # visual_time_step, input_seq,_ = attention_weights.shape
    # batch_feature_attention = np.zeros((visual_time_step, num_vars, num_vars))

    # for b in range(visual_time_step):
    #     for i in range(num_vars):
    #         for j in range(num_vars):
    #             batch_feature_attention[b, i, j] = np.dot(
    #                 attention_mean_heads[b, i], attention_mean_heads[b, j]
    #             )

    # # Step 3: Average across all batches to get a single feature-to-feature attention matrix
    # global_feature_attention = np.mean(batch_feature_attention, axis=0)  # Shape: (num_vars, num_vars)

    # # Step 4: Normalize the feature-to-feature attention matrix
    # min_value = global_feature_attention.min()
    # max_value = global_feature_attention.max()
    # normalized_attention = (global_feature_attention - min_value) / (max_value - min_value)

    # Step 5: Visualize the normalized feature-to-feature attention matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(attention_weights, cmap='plasma', interpolation='nearest')
    # plt.title(f"Normalized Feature-to-Feature Attention")
    # plt.xlabel("Key")
    # plt.ylabel("Query")

    # # Define feature names
    # feature_names = [
    # r"$\eta$", r"$\phi$", r"$\rho$", r"$\sigma$", r"$\alpha$",
    # r"$\eta_{t}$", r"$\eta_{s}$", r"$\eta_{r}$",
    # r"$\phi_{t}$", r"$\phi_{s}$", r"$\phi_{r}$",
    # r"$\rho_{t}$", r"$\rho_{s}$", r"$\rho_{r}$",
    # r"$\sigma_{t}$", r"$\sigma_{s}$", r"$\sigma_{r}$",
    # r"$\alpha_{t}$", r"$\alpha_{s}$", r"$\alpha_{r}$"
    # ]


    # # Ensure the number of features matches the names
    # # if len(feature_names) != num_vars:
    # #     raise ValueError("Number of feature names does not match num_vars.")

    # # plt.xticks(range(num_vars), labels=feature_names, rotation=90)
    # # plt.yticks(range(num_vars), labels=feature_names)

    # # Add colorbar
    # cbar = plt.colorbar()
    # cbar.set_label("Attention Weight")

    # plt.tight_layout()

    # # Save the figure
    # filename = f"VHF_{file_prefix}_feature_attention.png"
    # save_path = os.path.join(save_dir, filename)
    # plt.savefig(save_path)
    # plt.close()

    # print(f"Saved normalized feature-to-feature attention heatmap: {save_path}")
