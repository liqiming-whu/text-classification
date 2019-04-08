library("ggplot2")
library("Cairo")

base<-read.table("D:\\config\\base.txt",header=T)
att1<-read.table("D:\\config\\att1.txt",header=T)
att2<-read.table("D:\\config\\att2.txt",header=T)
att3<-read.table("D:\\config\\att3.txt",header=T)

dt<-merge(base,att1,by="x")
dt<-merge(dt,att2,by="x")
dt<-merge(dt,att3,by="x")

CairoPNG(file="D:\\config\\base.png",units="in",bg="white",width=5.5,height=5,dpi=300)

ba<-ggplot(base,aes(x,Base_CNN))+geom_point(alpha=0.4, size=1.0)+geom_smooth(method="loess",color="red")+
  labs(x="迭代次数",y="损失函数",title="Base_CNN")+
  theme_bw(base_size = 12, base_family = "Times") +
  theme(panel.grid=element_blank(),text = element_text(family = "STXihei"),plot.title = element_text(hjust = 0.5))
print(ba)
ggsave("D:\\config\\base.png",bg="white",width=5.5,height=5,dpi=300)
dev.off()

CairoPNG(file="D:\\config\\att1.png",units="in",bg="white",width=5.5,height=5,dpi=300)

a1<-ggplot(att1,aes(x,Att_CNN1))+geom_point(alpha=0.4, size=1.0)+geom_smooth(method="loess",color="red")+
  labs(x="迭代次数",y="损失函数",title="Att_CNN1")+
  theme_bw(base_size = 12, base_family = "Times") +
  theme(panel.grid=element_blank(),text = element_text(family = "STXihei"),plot.title = element_text(hjust = 0.5))
print(a1)
ggsave("D:\\config\\att1.png",bg="white",width=5.5,height=5,dpi=300)
dev.off()

CairoPNG(file="D:\\config\\att2.png",units="in",bg="white",width=5.5,height=5,dpi=300)

a2<-ggplot(att2,aes(x,Att_CNN2))+geom_point(alpha=0.4, size=1.0)+geom_smooth(method="loess",color="red")+
  labs(x="迭代次数",y="损失函数",title="Att_CNN2")+
  theme_bw(base_size = 12, base_family = "Times") +
  theme(panel.grid=element_blank(),text = element_text(family = "STXihei"),plot.title = element_text(hjust = 0.5))
print(a2)
ggsave("D:\\config\\att2.png",bg="white",width=5.5,height=5,dpi=300)
dev.off()

CairoPNG(file="D:\\config\\att3.png",units="in",bg="white",width=5.5,height=5,dpi=300)

a3<-ggplot(att3,aes(x,Att_CNN3))+geom_point(alpha=0.4, size=1.0)+geom_smooth(method="loess",color="red")+
  labs(x="迭代次数",y="损失函数",title="Att_CNN3")+
  theme_bw(base_size = 12, base_family = "Times") +
  theme(panel.grid=element_blank(),text = element_text(family = "STXihei"),plot.title = element_text(hjust = 0.5))
print(a3)
ggsave("D:\\config\\att3.png",bg="white",width=5.5,height=5,dpi=300)
dev.off()

CairoPNG(file="D:\\config\\convergence_speed.png",units="in",bg="white",width=5.5,height=5,dpi=300)

px<-ggplot(data=dt,aes(x=x))+geom_smooth(aes(y=Base_CNN,colour = "Base_CNN"),se=F)+
  geom_smooth(aes(y=Att_CNN1,colour = "Att_CNN1"),se=F)+
  geom_smooth(aes(y=Att_CNN2,colour = "Att_CNN2"),se=F)+
  geom_smooth(aes(y=Att_CNN3,colour = "Att_CNN3"),se=F)+
  scale_colour_manual(
    values = c("Base_CNN" = "red", "Att_CNN1" = "yellow", 
               "Att_CNN2" = "blue", "Att_CNN3" = "green"))+
  xlab("迭代次数")+ylab("损失函数")+
  theme_bw(base_size = 12, base_family = "Times") +
  theme(legend.title=element_blank(),panel.grid=element_blank(),legend.position = c(0.85, 0.85),text = element_text(family = "STXihei"))
print(px)
ggsave("D:\\config\\convergence_speed.png",bg="white",width=5.5,height=5,dpi=300)
dev.off()

