/// <reference types="react" />
import { DetailedHTMLProps, ImgHTMLAttributes } from 'react';

type ImageProps = DetailedHTMLProps<ImgHTMLAttributes<HTMLImageElement>, HTMLImageElement> & {
  priority?: boolean;
};

export function Image({ src, alt, width, height, className, priority, style, ...props }: ImageProps) {
  return (
    <img
      {...props}
      src={src}
      alt={alt}
      width={width}
      height={height}
      className={className}
      loading={priority ? "eager" : "lazy"}
      style={{
        width: width ? `${width}px` : 'auto',
        height: height ? `${height}px` : 'auto',
        ...style
      }}
    />
  );
} 