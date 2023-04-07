import logging

import torch.nn as nn

class Distiller(nn.Module):
    def init(
        self, 
        teacher, 
        teacher_tokenizer, 
        student, 
        student_tokenizer, 
        same_vocab=True,
        tokenizer_map=None,
        alpha=0.5,
    ):
        
        self.teacher = teacher
        self.teacher.requires_grad_(requires_grad=False)
        self.teacher_tokenizer = teacher_tokenizer
        
        self.student = student
        self.student_tokenizer = student_tokenizer
        self.same_vocab = same_vocab
        self.tokenizer_map = tokenizer_map
        
        self.alpha = alpha
        self.logger = logging.logger
        
        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, input):
        teacher_logits = self.teacher(input)
        student_logits = self.student(input)
        
        return teacher_logits, student_logits
    
    def loss_fn(self, student_logits, teacher_logits, teacher_outputs):
        student_loss = self.student_loss_fn(student_logits, teacher_outputs)
        distill_loss = self.distill_loss_fn(student_logits, teacher_logits)
        loss = self.alpha * student_loss + (1-self.alpha) * distill_loss
        
        self.logger.info(f"Total loss: {loss} | Student Loss: {student_loss} | Distil Loss: {distill_loss}")
        
        return loss
        
        
    