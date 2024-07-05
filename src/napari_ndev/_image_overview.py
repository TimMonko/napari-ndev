import matplotlib.pyplot as plt
from skimage import io

class ImageSet:
    def __init__(self, image_sets):
        self.image_sets = image_sets
        self.fig, self.axs = self._construct_subplot()
    
    def _construct_subplot(self):
        num_rows = len(self.image_sets)
        num_columns = max(len(image_set['images']) for image_set in self.image_sets)
        
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
        
        if num_rows == 1:
            axs = [axs]
        if num_columns == 1:
            axs = [[ax] for ax in axs]
        
        for row, image_set in enumerate(self.image_sets):
            images = image_set['images']
            colormaps = image_set.get('colormaps', ['gray'] * len(images))
            titles = image_set.get('titles', [''] * len(images))
            
            for col, image in enumerate(images):
                ax = axs[row][col]
                ax.imshow(image, cmap=colormaps[col])
                ax.set_title(titles[col])
                ax.axis('off')
        
        plt.tight_layout()
        return fig, axs

    def display(self):
        # Display the constructed subplot
        plt.show()

    def save(self, filepath):
        # Save the constructed subplot to a file
        # Note: This saves the entire figure as a single image
        self.fig.savefig(filepath)
        print(f"Saved figure to {filepath}")

# Example usage:
# image_sets = [
#     {'images': [img1, img2], 'colormaps': ['gray', 'viridis'], 'titles': ['First', 'Second']},
#     {'images': [img3], 'titles': ['Third']}
# ]
# img_set = ImageSet(image_sets)
# img_set.display()
# img_set.save('path/to/save/overview.png')