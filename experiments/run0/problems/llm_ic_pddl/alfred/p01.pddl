(define (problem move-bowl-to-shelf)
  (:domain household)
  (:objects 
      bowl - object
      shelf - location
      left right - hand
  )
  (:init 
    (found bowl)
    (handempty left)
    (handempty right)
    (on bowl shelf)
    (clean bowl)
  )
  (:goal
    (on bowl shelf)
  )
)