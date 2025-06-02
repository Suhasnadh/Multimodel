import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import BertModel

class MultimodalClassifier(nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()

        # Image Encoder (ResNet50)
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        # Text Encoder (BERT)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)  # BERT's output size is 768

        # Combined Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # 256 (text) + 256 (image)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)     # Output: 2 classes (Fake / Real)
        )

    def forward(self, image, input_ids, attention_mask):
        # Image Features
        img_feat = self.cnn(image)

        # Text Features from BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_fc(bert_out.pooler_output)

        # Combine text and image features
        combined = torch.cat((img_feat, text_feat), dim=1)

        # Classification
        return self.classifier(combined)
