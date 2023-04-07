import torch.nn as nn

class Distiller(nn.Module):
    def init(
        self, 
        teacher, 
        student, 
        teacher_tokenizer, 
        student_tokenizer, 
        same_vocab=True,
        vocab_prob_map=None,
        alpha=0.5,
    ):
        
        self.teacher = teacher
        self.teacher.requires_grad_(requires_grad=False)
        self.teacher_tokenizer = teacher_tokenizer
        
        self.student = student
        self.student_tokenizer = student_tokenizer
        self.same_vocab = same_vocab
        self.vocab_prob_map = vocab_prob_map
        
        self.alpha = alpha
        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, teacher_input, student_input):
        teacher_outputs = self.teacher(teacher_input)
        student_outputs = self.student(student_input)
        
        return teacher_outputs, student_outputs
    
    def loss_fn(self, teacher_logits, teacher_preds, student_logits, student_preds):
        student_loss = self.student_loss_fn(student_preds, teacher_preds)
        distill_loss = self.distill_loss_fn(student_logits, teacher_logits)
        loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        
        return loss, student_loss, distill_loss
        
        
    